import os
import logging
import base64
import sqlite3
import re
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests
import openai
import feedparser
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL')
GMAIL_CREDENTIALS_PATH = os.getenv('GMAIL_CREDENTIALS_PATH', 'credentials.json')
MAX_EMAILS = int(os.getenv('MAX_EMAILS', 50))  # Increased for more content
DAYS_BACK = int(os.getenv('DAYS_BACK', 7))
# Keywords to identify newsletters, all stored lowercase
NEWSLETTER_KEYWORDS = [kw.strip().lower() for kw in os.getenv(
    'NEWSLETTER_KEYWORDS',
    'AI,artificial intelligence,machine learning,newsletter,digest,weekly,update'
).split(',')]
EXCLUDED_KEYWORDS = [kw.strip().lower() for kw in os.getenv(
    'EXCLUDED_KEYWORDS',
    'unsubscribe,spam,promotion,offer,sale,discount'
).split(',')]
NEWSLETTER_SENDERS = os.getenv('NEWSLETTER_SENDERS', '').split(',')
EXCLUDED_SENDERS = os.getenv('EXCLUDED_SENDERS', '').split(',')
DEBUG = os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes')
GMAIL_LABEL = os.getenv('GMAIL_LABEL', 'ai-newsletters')


def setup_logging(debug: bool = False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(Path('ai_newsletter.log'), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    class NoEmojiFilter(logging.Filter):
        def filter(self, record):
            record.msg = ''.join(c for c in str(record.msg) if ord(c) < 128)
            return True
    console_handler.addFilter(NoEmojiFilter())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_db_connection(db_path: Path):
    return sqlite3.connect(str(db_path))


class Config:
    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.newsapi_key = NEWSAPI_KEY
        self.smtp_user = SMTP_USER
        self.smtp_password = SMTP_PASSWORD
        self.to_email = TO_EMAIL
        self.gmail_credentials_path = GMAIL_CREDENTIALS_PATH
        self.max_emails = MAX_EMAILS
        self.days_back = DAYS_BACK
        self.newsletter_keywords = NEWSLETTER_KEYWORDS
        self.excluded_keywords = EXCLUDED_KEYWORDS
        self.newsletter_senders = NEWSLETTER_SENDERS
        self.excluded_senders = EXCLUDED_SENDERS
        self.db_path = Path('newsletter_db.sqlite')
        self.gmail_label = GMAIL_LABEL


class EmailFilter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Premium media sources - prioritize these
        self.premium_sources = [
            'wsj.com', 'bloomberg.com', 'ft.com', 'reuters.com', 'theinformation.com',
            'wall street journal', 'bloomberg', 'financial times', 'reuters', 
            'new york times', 'washington post', 'techcrunch', 'axios', 'cnbc'
        ]
        
    def is_valid_newsletter(self, email_data: Dict) -> bool:
        subject = email_data.get('subject', '').lower()
        sender = email_data.get('sender', '').lower()
        body = email_data.get('body', '').lower()
        
        # Check exclusions first
        if any(ex in sender for ex in self.config.excluded_senders):
            self.logger.debug(f"Excluded sender: {sender}")
            return False
            
        # Strong exclusion for non-AI content
        exclusion_terms = [
            # Financial (unless AI-related)
            'stock', 'shares', 'etf', 'investment', 'trading', 'nasdaq', 'nyse', 'market cap',
            # News categories
            'politics', 'election', 'climate', 'weather', 'sports', 'entertainment', 'celebrity',
            'food', 'recipe', 'milk', 'pecan', 'cooking', 'restaurant', 'cuisine',
            # General news
            'war', 'crisis', 'breaking news', 'conspiracy theory', 'cia', 'oswald', 'kennedy',
            'obesity', 'health', 'medical', 'hospital', 'quality of life', 'tourism',
            # More specific exclusions
            'kenya', 'russia', 'north korea', 'storms', 'flooding', 'subway', 'highways',
            'epstein', 'cuomo', 'netanyahu', 'nukes', 'nuclear', 'vitamin d tracker'
        ]
        
        if any(term in subject or term in body for term in exclusion_terms):
            # Allow only if it's explicitly AI-related
            ai_integration_terms = [
                'ai', 'artificial intelligence', 'machine learning', 'neural network', 
                'deep learning', 'openai', 'chatgpt', 'claude', 'gpt', 'llm'
            ]
            if not any(ai_term in subject or ai_term in body for ai_term in ai_integration_terms):
                self.logger.debug(f"Excluded non-AI content: {subject}")
                return False
        
        # Calculate AI relevance score
        ai_score = self._calculate_ai_relevance(subject, body, sender)
        
        # Higher threshold to ensure AI relevance
        is_premium = self._is_premium_source(sender)
        threshold = 3 if is_premium else 4
        is_valid = ai_score >= threshold
        
        self.logger.debug(f"Email '{subject}' from {'premium' if is_premium else 'regular'} source - AI score: {ai_score} -> Valid: {is_valid}")
        return is_valid
    
    def _is_premium_source(self, sender: str) -> bool:
        """Check if sender is from a premium media source"""
        return any(source in sender for source in self.premium_sources)
    
    def _calculate_ai_relevance(self, subject: str, body: str, sender: str = '') -> int:
        score = 0
        
        # High-value AI terms
        high_value_terms = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'llm', 'gpt', 'openai', 'claude']
        for term in high_value_terms:
            if term in subject:
                score += 3
            if term in body:
                score += 2
        
        # Medium-value AI terms  
        medium_value_terms = ['ai ', 'ml ', 'chatbot', 'automation', 'algorithm', 'data science']
        for term in medium_value_terms:
            if term in subject:
                score += 2
            if term in body:
                score += 1
                
        # Newsletter indicators
        newsletter_terms = ['newsletter', 'digest', 'weekly', 'daily', 'briefing', 'roundup']
        if any(term in subject for term in newsletter_terms):
            score += 1
        
        # Premium source bonus
        if self._is_premium_source(sender):
            score += 1
            
        return score


class GmailService:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.service = self._initialize_gmail_service()

    def _initialize_gmail_service(self):
        creds = None
        creds_path = Path(self.config.gmail_credentials_path)
        token_path = creds_path.with_name('token.json')
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(
                token_path,
                ['https://www.googleapis.com/auth/gmail.readonly',
                 'https://www.googleapis.com/auth/gmail.send']
            )
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    creds_path,
                    ['https://www.googleapis.com/auth/gmail.readonly',
                     'https://www.googleapis.com/auth/gmail.send']
                )
                creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token_fp:
                    token_fp.write(creds.to_json())
        self.logger.info("Gmail service initialized successfully")
        return build('gmail', 'v1', credentials=creds)

    def _get_email_data(self, message_id: str) -> Dict:
        try:
            msg = self.service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
            date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
            body = self._get_body(msg['payload'])
            return {'id': message_id, 'subject': subject, 'sender': sender, 'body': body, 'date': date}
        except Exception as e:
            self.logger.error(f"Error getting email data: {e}")
            return {}

    def _get_body(self, payload) -> str:
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                    return base64.urlsafe_b64decode(
                        part['body']['data']
                    ).decode('utf-8', errors='ignore')
        elif payload.get('body', {}).get('data'):
            return base64.urlsafe_b64decode(
                payload['body']['data']
            ).decode('utf-8', errors='ignore')
        return ''

    def _fetch_from_label(self, email_filter: EmailFilter, processed: Set[str]) -> List[Dict]:
        items = []
        try:
            labels_res = self.service.users().labels().list(userId='me').execute()
            all_labels = [lbl['name'] for lbl in labels_res.get('labels', [])]
            self.logger.info(f"Available Gmail labels: {', '.join(all_labels[:10])}...")  # Show first 10
            
            label_id = self._find_label_flexible(labels_res.get('labels', []), self.config.gmail_label)
            if not label_id:
                self.logger.warning(f"Label '{self.config.gmail_label}' not found. Available labels: {all_labels}")
                return items
            
            self.logger.info(f"Found label '{self.config.gmail_label}' with ID: {label_id}")

            # Fetch messages from the last N days (configurable)
            start_date = datetime.now().date() - timedelta(days=self.config.days_back)
            end_date = datetime.now().date()
            q = f"after:{start_date.strftime('%Y/%m/%d')} before:{end_date.strftime('%Y/%m/%d')}"
            self.logger.info(f"Searching Gmail label '{self.config.gmail_label}' from {start_date} to {end_date}")
            result = self.service.users().messages().list(
                userId='me',
                labelIds=[label_id],
                q=q,
                maxResults=self.config.max_emails
            ).execute()
            msgs = result.get('messages', [])
            self.logger.info(f"Found {len(msgs)} messages in label '{self.config.gmail_label}' from {start_date} to {end_date}")
            for m in msgs:
                if m['id'] in processed:
                    continue
                data = self._get_email_data(m['id'])
                if data:
                    self.logger.debug(f"Processing email: {data.get('subject', 'No Subject')}")
                    if email_filter.is_valid_newsletter(data):
                        items.append(data)
                        processed.add(m['id'])
                        self.logger.info(f"Added newsletter: {data.get('subject', 'No Subject')}")
                    else:
                        processed.add(m['id'])
                        self.logger.debug(f"Filtered out: {data.get('subject', 'No Subject')}")
                else:
                    self.logger.warning(f"Failed to get email data for message {m['id']}")
        except Exception as e:
            self.logger.error(f"Error fetching from label: {e}")
        return items

    def send_email(self, html: str, subject: str):
        msg = MIMEText(html, 'html')
        msg['to'] = self.config.to_email
        msg['from'] = self.config.smtp_user
        msg['subject'] = subject

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        try:
            self.service.users().messages().send(userId='me', body={'raw': raw}).execute()
            self.logger.info("Email sent successfully")
        except Exception as e:
            self.logger.error(f"Send error: {e}")

    def _fetch_by_content_analysis(self, email_filter: EmailFilter, processed: Set[str]) -> List[Dict]:
        items = []
        try:
            since = datetime.now() - timedelta(days=self.config.days_back)
            
            # Comprehensive search strategies for maximum coverage
            search_queries = [
                # Premium media sources
                f"after:{since.strftime('%Y/%m/%d')} (from:wsj.com OR from:bloomberg.com OR from:ft.com OR from:reuters.com OR from:nytimes.com OR from:washingtonpost.com)",
                # Tech publications
                f"after:{since.strftime('%Y/%m/%d')} (from:techcrunch.com OR from:theverge.com OR from:wired.com OR from:arstechnica.com OR from:engadget.com)",
                # AI companies and research
                f"after:{since.strftime('%Y/%m/%d')} (from:openai.com OR from:anthropic.com OR from:deepmind.com OR from:microsoft.com OR from:google.com)",
                # AI-focused search with variations
                f"after:{since.strftime('%Y/%m/%d')} (AI OR \"artificial intelligence\" OR \"machine learning\" OR \"deep learning\" OR \"neural network\")",
                f"after:{since.strftime('%Y/%m/%d')} (OpenAI OR ChatGPT OR Claude OR GPT OR \"large language model\" OR LLM)",
                f"after:{since.strftime('%Y/%m/%d')} (\"generative AI\" OR \"AI safety\" OR \"AI ethics\" OR \"AI regulation\" OR \"AI governance\")",
                # Newsletter and content formats
                f"after:{since.strftime('%Y/%m/%d')} (newsletter OR digest OR briefing OR roundup OR update) (AI OR tech OR technology OR innovation)",
                # Research and academic
                f"after:{since.strftime('%Y/%m/%d')} (research OR study OR paper OR breakthrough) (AI OR \"artificial intelligence\" OR ML)",
                # Industry and business
                f"after:{since.strftime('%Y/%m/%d')} (startup OR funding OR investment OR acquisition) (AI OR \"artificial intelligence\")",
                # Tools and products
                f"after:{since.strftime('%Y/%m/%d')} (tool OR platform OR service OR product) (AI OR \"artificial intelligence\" OR \"machine learning\")"
            ]
            
            all_msg_ids = set()
            
            for query in search_queries:
                try:
                    msgs = self.service.users().messages().list(
                        userId='me', q=query, maxResults=self.config.max_emails
                    ).execute().get('messages', [])
                    
                    for msg in msgs:
                        all_msg_ids.add(msg['id'])
                        
                    self.logger.debug(f"Query '{query[:50]}...' found {len(msgs)} messages")
                except Exception as e:
                    self.logger.warning(f"Search query failed: {e}")
                    continue
            
            self.logger.info(f"Found {len(all_msg_ids)} total unique messages via content analysis")
            
            for msg_id in all_msg_ids:
                if msg_id in processed:
                    continue
                    
                data = self._get_email_data(msg_id)
                if data:
                    # More flexible date filtering for premium sources
                    try:
                        recv_dt = parsedate_to_datetime(data['date'])
                        is_premium = email_filter._is_premium_source(data.get('sender', ''))
                        
                        # Allow wider date range for premium sources
                        max_days_back = self.config.days_back if is_premium else 2
                        if recv_dt.date() < (datetime.now().date() - timedelta(days=max_days_back)):
                            continue
                    except Exception:
                        pass  # Include if date parsing fails
                    
                    if email_filter.is_valid_newsletter(data):
                        items.append(data)
                        processed.add(msg_id)
                        self.logger.info(f"Added from content analysis: {data.get('subject', 'No Subject')}")
                    else:
                        processed.add(msg_id)
                        
        except Exception as e:
            self.logger.error(f"Error fetching by content: {e}")
        return items

    def fetch_newsletters(self, email_filter: EmailFilter) -> List[Dict]:
        processed = set()
        newsletters = []
        
        # Try label-based approach first
        newsletters.extend(self._fetch_from_label(email_filter, processed))
        
        # Always try content analysis for comprehensive coverage
        newsletters.extend(self._fetch_by_content_analysis(email_filter, processed))
        
        # Remove duplicates based on subject similarity
        unique_newsletters = self._deduplicate_by_subject(newsletters)
        
        self.logger.info(f"Gmail: Found {len(unique_newsletters)} unique newsletters from {len(newsletters)} total")
        return unique_newsletters[:self.config.max_emails]
    
    def _deduplicate_by_subject(self, newsletters: List[Dict]) -> List[Dict]:
        """Remove duplicate newsletters based on subject similarity"""
        seen_subjects = set()
        unique_newsletters = []
        
        for newsletter in newsletters:
            subject_key = newsletter.get('subject', '').lower().strip()
            # Simple deduplication - could be enhanced
            if subject_key not in seen_subjects and subject_key:
                seen_subjects.add(subject_key)
                unique_newsletters.append(newsletter)
        
        return unique_newsletters

    def _normalize_label_name(self, label_name: str) -> str:
        """Normalize label name for flexible matching"""
        return label_name.lower().replace(' ', '').replace('-', '').replace('_', '')

    def _find_label_flexible(self, labels: List[Dict], target_label: str) -> str:
        """Find Gmail label with flexible matching for spaces, hyphens, and case"""
        if not labels or not target_label:
            return None
        
        target_normalized = self._normalize_label_name(target_label)
        
        # First try exact match (case-insensitive)
        for label in labels:
            if label['name'].lower() == target_label.lower():
                return label['id']
        
        # Then try normalized matching (remove spaces, hyphens, underscores)
        for label in labels:
            if self._normalize_label_name(label['name']) == target_normalized:
                self.logger.info(f"Found label '{label['name']}' matching '{target_label}' (flexible match)")
                return label['id']
        
        # Finally try partial matching
        for label in labels:
            if target_normalized in self._normalize_label_name(label['name']) or \
               self._normalize_label_name(label['name']) in target_normalized:
                self.logger.info(f"Found label '{label['name']}' partially matching '{target_label}'")
                return label['id']
        
        return None



class NewsAPIService:
    def __init__(self, config: Config):
        self.config=config;self.logger=logging.getLogger(__name__);self.enabled=bool(config.newsapi_key)
        if not self.enabled: self.logger.warning("NewsAPI disabled: no API key")
    @retry(stop=stop_after_attempt(3),wait=wait_fixed(1),retry=retry_if_exception_type(requests.RequestException))
    def fetch_ai_news(self)->List[Dict]:
        if not self.enabled: return []
        cache=Path(f"newsapi_cache_{datetime.now().strftime('%Y-%m-%d')}.json")
        if cache.exists(): return json.loads(cache.read_text())
        
        # Fetch from premium sources first
        all_items = []
        
        # Multiple API calls for broader coverage
        api_queries = [
            # Premium sources
            {
                'q':'("artificial intelligence" OR "machine learning" OR AI OR "OpenAI" OR "ChatGPT") -stock -shares -earnings -investment',
                'domains': "wsj.com,bloomberg.com,reuters.com,ft.com,nytimes.com,washingtonpost.com,techcrunch.com,axios.com,cnbc.com",
                'pageSize': 20, 'is_premium': True
            },
            # AI-focused tech sources
            {
                'q':'("artificial intelligence" OR "machine learning" OR "deep learning" OR "neural network" OR "OpenAI" OR "ChatGPT" OR "Claude" OR "LLM")',
                'domains': "techcrunch.com,theverge.com,wired.com,arstechnica.com,venturebeat.com,engadget.com",
                'pageSize': 15, 'is_premium': False
            },
            # General comprehensive search
            {
                'q':'("artificial intelligence" OR "machine learning" OR "deep learning" OR "OpenAI" OR "ChatGPT" OR "Claude" OR "LLM" OR "AI tools" OR "AI research") -stock -shares -earnings -investment -ETF -financial',
                'pageSize': 25, 'is_premium': False
            },
            # Emerging AI topics
            {
                'q':'("generative AI" OR "AI safety" OR "AI regulation" OR "AI ethics" OR "AI governance" OR "AGI" OR "multimodal AI")',
                'pageSize': 15, 'is_premium': False
            }
        ]
        
        url="https://newsapi.org/v2/everything"
        
        # Execute multiple queries for comprehensive coverage
        for i, query_config in enumerate(api_queries):
            try:
                params = {
                    'q': query_config['q'],
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'from': (datetime.now()-timedelta(days=self.config.days_back)).strftime('%Y-%m-%d'),
                    'pageSize': query_config['pageSize'],
                    'apiKey': self.config.newsapi_key
                }
                
                # Add domains if specified
                if 'domains' in query_config:
                    params['domains'] = query_config['domains']
                
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                articles = data.get('articles', [])
                processed_articles = self._process_articles(articles, is_premium=query_config.get('is_premium', False))
                all_items.extend(processed_articles)
                
                self.logger.info(f"Query {i+1}: Found {len(processed_articles)} articles from {len(articles)} total")
                
            except Exception as e:
                self.logger.warning(f"NewsAPI query {i+1} failed: {e}")
                continue
        
        # Remove duplicates and cache
        unique_items = self._deduplicate_articles(all_items)
        cache.write_text(json.dumps(unique_items))
        premium_count = len([i for i in unique_items if 'Premium' in i.get('sender', '')])
        self.logger.info(f"Fetched {len(unique_items)} articles ({premium_count} from premium sources)")
        return unique_items
    
    def _process_articles(self, articles: List[Dict], is_premium: bool = False) -> List[Dict]:
        """Process articles with enhanced filtering"""
        items = []
        excluded_terms = ['stock', 'shares', 'earnings', 'etf', 'investment', 'financial', 'trading', 'market cap', 'nasdaq', 'nyse']
        
        for art in articles:
            if art.get('title') and art.get('url'):
                title_lower = art['title'].lower()
                desc_lower = (art.get('description') or '').lower()
                
                # Skip if contains excluded financial terms (unless premium and AI-focused)
                if any(term in title_lower or term in desc_lower for term in excluded_terms):
                    if not is_premium:
                        continue
                    # For premium sources, only allow if strongly AI-related
                    ai_terms = ['artificial intelligence', 'ai ', 'openai', 'chatgpt', 'machine learning']
                    if not any(term in title_lower for term in ai_terms):
                        continue
                
                # For non-premium, ensure strong AI relevance
                if not is_premium:
                    ai_terms = ['artificial intelligence', 'machine learning', 'ai ', 'openai', 'chatgpt', 'claude', 'llm', 'neural', 'deep learning']
                    if not any(term in title_lower or term in desc_lower for term in ai_terms):
                        continue
                
                source_name = art.get('source', {}).get('name', 'NewsAPI')
                if is_premium:
                    source_name = f"{source_name} (Premium)"
                    
                items.append({
                    'id': f"newsapi_{hashlib.md5(art['url'].encode()).hexdigest()}",
                    'subject': art['title'],
                    'sender': source_name,
                    'body': f"{art.get('description', '')} ({art['url']})",
                    'date': art.get('publishedAt', '')
                })
        return items
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title_key = article['subject'].lower().strip()
            # Simple deduplication - could be enhanced with fuzzy matching
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles


class RSSFeedService:
    """RSS feed service for additional AI content"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Major AI RSS feeds
        self.ai_feeds = [
            'https://feeds.feedburner.com/venturebeat/SZYF',  # VentureBeat AI
            'https://techcrunch.com/tag/artificial-intelligence/feed/',  # TechCrunch AI
            'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml',  # The Verge AI
            'https://www.wired.com/feed/category/business/artificial-intelligence/latest/rss',  # Wired AI
            'https://feeds.feedburner.com/oreilly/radar',  # O'Reilly Radar
            'https://ai.googleblog.com/feeds/posts/default',  # Google AI Blog
            'https://openai.com/blog/rss.xml',  # OpenAI Blog
            'https://blogs.microsoft.com/ai/feed/',  # Microsoft AI Blog
            'https://research.facebook.com/blog/feed/',  # Meta Research
        ]
    
    def fetch_rss_content(self) -> List[Dict]:
        """Fetch content from RSS feeds"""
        all_items = []
        
        for feed_url in self.ai_feeds:
            try:
                self.logger.debug(f"Fetching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Limit to 5 items per feed
                    # Check if entry is recent
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                            if pub_date.date() < (datetime.now().date() - timedelta(days=self.config.days_back)):
                                continue
                    except:
                        pass  # Include if date parsing fails
                    
                    # Extract content
                    title = entry.get('title', 'No Title')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')
                    source = feed.feed.get('title', 'RSS Feed')
                    
                    # Filter for AI relevance
                    if self._is_ai_relevant(title, summary):
                        all_items.append({
                            'id': f"rss_{hashlib.md5(link.encode()).hexdigest()}",
                            'subject': title,
                            'sender': f"{source} (RSS)",
                            'body': f"{summary[:200]}... ({link})",
                            'date': entry.get('published', '')
                        })
                
                self.logger.debug(f"RSS feed {feed_url} processed: {len([i for i in all_items if feed_url in str(i)])} items")
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                continue
        
        self.logger.info(f"Fetched {len(all_items)} items from RSS feeds")
        return all_items
    
    def _is_ai_relevant(self, title: str, summary: str) -> bool:
        """Check if RSS content is AI-relevant"""
        text = f"{title} {summary}".lower()
        ai_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 
            'neural network', 'ai ', 'openai', 'chatgpt', 'claude', 'llm',
            'generative ai', 'ai model', 'ai research', 'ai tools'
        ]
        return any(term in text for term in ai_terms)


class LLMProcessor:
    def __init__(self, config: Config):
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.config = config
        self.logger = logging.getLogger(__name__)
    def generate_summary(self, newsletters: List[Dict]) -> str:
        """Generate executive summary of key AI developments"""
        if not newsletters:
            return "No significant AI developments found today."
        
        # Create a condensed content sample
        content_sample = "\n".join([
            f"- {nl['subject']}: {nl['body'][:150]}..." 
            for nl in newsletters[:8]  # Use top 8 items
        ])
        
        summary_prompt = f"""
Based on today's AI news, provide a brief executive summary (2-3 sentences) of the most significant AI developments and trends.

Focus on:
- Major breakthroughs or announcements
- Important policy/regulatory changes  
- Significant industry developments
- Key research findings

AI News Today:
{content_sample}
"""
        
        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an AI industry analyst. Provide concise, insightful summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Today's AI developments span {len(newsletters)} items covering tools, research, and industry updates."

    def categorize_newsletters(self,newsletters:List[Dict])->Dict:
        if not newsletters: return {}
        
        # Temporarily force fallback categorization for testing
        self.logger.info("🔄 Using improved keyword-based categorization...")
        return self._create_fallback_categories(newsletters)
        
        content="\n".join(f"Newsletter: {nl['subject']} (Source: {nl['sender']})\nContent: {nl['body']}" for nl in newsletters)
        prompt=f"""
You are an AI news analyst. Categorize ONLY AI-related content. Distribute items across multiple categories.

SPECIFIC CATEGORIZATION RULES:
- AI Regulation & Policy: Government contracts, defense contracts, EU policies, regulations, compliance
- AI Tools & Products: Product launches, new AI tools, platform releases, consumer applications  
- AI Research & Development: Academic research, technical papers, breakthrough studies, open source
- AI Business & Industry: Company acquisitions, hiring, funding, partnerships, earnings of AI companies
- AI Ethics & Safety: Safety concerns, bias issues, responsible AI, threat detection

IMPORTANT: 
- Distribute items across ALL relevant categories
- Do NOT put everything in "Other AI News"
- Each item should go in the MOST SPECIFIC category
- If unsure, use broader categories like "AI Business & Industry"

Return VALID JSON with category names as keys and arrays of item descriptions as values.
Format each item as: "Item description (Source: Source Name)"

Content to analyze:
{content}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an AI news analyst. Always return valid JSON. NEVER use markdown formatting. Return only the JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            text = resp.choices[0].message.content.strip()
            self.logger.info(f"🤖 LLM categorization response length: {len(text)} chars")
            self.logger.debug(f"🤖 LLM response preview: {text[:200]}...")
            
            # Clean up the response to ensure valid JSON
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            try:
                result = json.loads(text)
                # Validate the result structure
                if isinstance(result, dict):
                    self.logger.info(f"✅ LLM categorization successful: {len(result)} categories")
                    return result
                else:
                    self.logger.warning("LLM returned non-dict result, using fallback")
                    return self._parse_fallback_response(text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parsing failed: {e}")
                self.logger.debug(f"Failed JSON text: {text[:500]}...")
                return self._parse_fallback_response(text)
                
        except Exception as e:
            self.logger.error(f"LLM categorization error: {e}")
            self.logger.info("🔄 Using fallback categorization...")
            return self._create_fallback_categories(newsletters)
    
    def _create_fallback_categories(self, newsletters: List[Dict]) -> Dict[str, List[str]]:
        """Create fallback categories when LLM fails"""
        categories = {
            "AI Fraud & Scams": [],
            "AI Regulation & Policy": [],
            "AI Tools & Products": [],
            "AI Research & Development": [],
            "AI Business & Industry": [],
            "AI Ethics & Safety": [],
            "Other AI News": []
        }
        
        for nl in newsletters:
            # Create clickable link if URL is available
            url = self._extract_url_from_body(nl.get('body', ''))
            if url:
                item_text = f"<a href=\"{url}\" target=\"_blank\">{nl['subject']}</a> (Source: {nl['sender']})"
            else:
                item_text = f"{nl['subject']} (Source: {nl['sender']})"
            subject_lower = nl['subject'].lower()
            
            # PRIORITY: Fraud detection (highest priority)
            fraud_terms = ['fraud', 'scam', 'deepfake', 'deepfakes', 'voice cloning', 'fake', 'synthetic media', 
                          'identity theft', 'impersonation', 'misinformation', 'disinformation', 'catfish', 
                          'phishing', 'criminal', 'deceptive', 'manipulation']
            if any(term in subject_lower for term in fraud_terms):
                categories["AI Fraud & Scams"].append(item_text)
            
            # Other categorization rules
            elif any(term in subject_lower for term in ['government', 'defense', 'contract', 'policy', 'regulation', 'eu', 'law']):
                categories["AI Regulation & Policy"].append(item_text)
            elif any(term in subject_lower for term in ['acquisition', 'acquires', 'funding', 'partnership', 'earnings', 'stock', 'investment', 'company']):
                categories["AI Business & Industry"].append(item_text)
            elif any(term in subject_lower for term in ['product', 'launch', 'tool', 'platform', 'app', 'software', 'copilot', 'grok', 'chatgpt']):
                categories["AI Tools & Products"].append(item_text)
            elif any(term in subject_lower for term in ['research', 'study', 'paper', 'academic', 'breakthrough', 'model', 'llm']):
                categories["AI Research & Development"].append(item_text)
            elif any(term in subject_lower for term in ['safety', 'ethics', 'bias', 'danger', 'threat', 'risk']):
                categories["AI Ethics & Safety"].append(item_text)
            else:
                categories["Other AI News"].append(item_text)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        total_items = sum(len(items) for items in categories.values())
        self.logger.info(f"📂 Fallback categorization: {total_items} items across {len(categories)} categories")
        return categories
    def _parse_fallback_response(self,text:str)->Dict[str,List[str]]:
        cats={};curr=None
        for line in text.splitlines():
            l=line.strip()
            if not l: continue
            if not l.startswith(('-', '•')):
                curr=l.rstrip(':');cats[curr]=[]
            elif curr: cats[curr].append(l.lstrip('-• '))
        return cats

    def assess_risk_levels(self, newsletters: List[Dict]) -> Dict[str, str]:
        """Assess risk levels for newsletter items"""
        if not newsletters:
            return {}
        
        # Prepare content for risk assessment
        content_for_risk = []
        for nl in newsletters:
            content_for_risk.append(f"Item: {nl['subject']}\nSource: {nl['sender']}\nContent: {nl['body'][:300]}")
        
        content_text = "\n\n---\n\n".join(content_for_risk)
        
        risk_prompt = f"""
Assess the security and threat risk level for each AI news item. Focus on immediate threat potential and security implications for organizations and individuals.

RISK LEVELS AND THREAT CATEGORIES:

HIGH RISK 🔴 (Immediate Security Threats):
- Fraud & Scams: Deepfakes, voice cloning, AI-generated fraud, identity theft, investment scams
- Active Threats: AI-powered attacks, security breaches, malware, phishing campaigns
- Weaponization: AI used for misinformation, election interference, propaganda, social engineering
- Criminal Exploitation: AI tools being used for illegal activities, criminal operations
- Critical Vulnerabilities: Security flaws in AI systems, data breaches, system compromises

MEDIUM RISK 🟡 (Business & Strategic Impact):
- Corporate Events: M&A, partnerships, major funding, strategic alliances, leadership changes
- Regulatory Changes: Policy updates, compliance requirements, government regulations
- Competitive Intelligence: Market disruptions, industry shifts, competitive developments
- Government Contracts: Defense deals, public sector AI adoption, policy implementations
- Infrastructure Changes: Major platform updates, system deployments, technology shifts

LOW RISK 🟢 (Operational & Educational):
- Technical Improvements: Feature updates, performance enhancements, bug fixes, optimizations
- Education & Training: Tutorials, courses, documentation, best practices, skill development
- Research & Academia: Papers, studies, theoretical work, academic discussions
- General Content: News commentary, opinion pieces, industry analysis, trend discussions
- Product Announcements: Minor releases, beta features, tool launches without security implications

Prioritize security threats over business impact. Consider the immediate actionable threat to users and organizations.

Return VALID JSON mapping item titles to risk levels:
{{"Item Title 1": "HIGH", "Item Title 2": "MEDIUM", "Item Title 3": "LOW"}}

Content to assess:
{content_text}
"""
        
        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity and AI safety analyst specializing in threat assessment and security risk evaluation. Focus on identifying security threats, fraud, and malicious AI use. Always return valid JSON."},
                    {"role": "user", "content": risk_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            text = resp.choices[0].message.content.strip()
            
            # Clean up the response
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            try:
                risk_levels = json.loads(text)
                if isinstance(risk_levels, dict):
                    return risk_levels
                else:
                    self.logger.warning("Risk assessment returned non-dict, using fallback")
                    return self._fallback_risk_assessment(newsletters)
            except json.JSONDecodeError:
                self.logger.warning("Risk assessment JSON parsing failed, using fallback")
                return self._fallback_risk_assessment(newsletters)
                
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return self._fallback_risk_assessment(newsletters)
    
    def _fallback_risk_assessment(self, newsletters: List[Dict]) -> Dict[str, str]:
        """Fallback risk assessment based on keywords"""
        risk_levels = {}
        
        high_risk_keywords = [
            # Security threats and fraud
            'fraud', 'scam', 'deepfake', 'deepfakes', 'voice cloning', 'synthetic media', 'fake', 'cloned',
            'identity theft', 'impersonation', 'misinformation', 'disinformation', 'manipulation',
            'attack', 'breach', 'hack', 'threat', 'malicious', 'criminal', 'exploitation',
            'security warning', 'alert', 'vulnerable', 'compromised', 'malware', 'phishing',
            'election interference', 'propaganda', 'ai-powered attack'
        ]
        medium_risk_keywords = [
            # Business and industry impact
            'acquisition', 'merger', 'partnership', 'funding', 'investment', 'ipo', 'contract',
            'regulation', 'policy', 'compliance', 'government', 'defense', 'military',
            'competition', 'market', 'industry', 'disruption', 'strategic', 'deal',
            'hired', 'poached', 'acquires', 'buys', 'earnings', 'revenue'
        ]
        
        for nl in newsletters:
            text = f"{nl['subject']} {nl['body']}".lower()
            
            # PRIORITY: AI Fraud & Scams - Always HIGH risk
            fraud_indicators = [
                'deepfake', 'deepfakes', 'voice cloning', 'voice clone', 'fraud', 'scam', 'fraudulent',
                'fake news', 'synthetic media', 'identity theft', 'impersonation', 'catfish',
                'misinformation', 'disinformation', 'manipulation', 'deceptive', 'criminal',
                'phishing', 'social engineering', 'investment scam', 'ponzi', 'pyramid scheme',
                'romance scam', 'ai-generated lies', 'fake identity', 'synthetic voice'
            ]
            
            if any(indicator in text for indicator in fraud_indicators):
                risk_levels[nl['subject']] = "HIGH"
            
            # Other high-risk security patterns
            elif any(phrase in text for phrase in ['ai attack', 'security breach', 'hack', 'malware', 'vulnerable', 'compromised']):
                risk_levels[nl['subject']] = "HIGH"
            elif any(keyword in text for keyword in high_risk_keywords):
                risk_levels[nl['subject']] = "HIGH"
            elif any(keyword in text for keyword in medium_risk_keywords):
                risk_levels[nl['subject']] = "MEDIUM"
            else:
                risk_levels[nl['subject']] = "LOW"
        
        return risk_levels

    def format_report(self, newsletters: List[Dict], categories: Dict, summary: str = None, risk_levels: Dict[str, str] = None) -> str:
        date_str = datetime.now().strftime('%Y-%m-%d')
        total_items = sum(len(items) for items in categories.values()) if categories else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily AI Newsletter Report - {date_str}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .category {{ background: white; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .category h2 {{ color: #667eea; margin-top: 0; font-size: 20px; border-bottom: 2px solid #f1f3f4; padding-bottom: 10px; }}
        .category ul {{ margin: 0; padding-left: 20px; }}
        .category li {{ margin: 8px 0; }}
        .category a {{ color: #667eea; text-decoration: none; font-weight: 500; }}
        .category a:hover {{ text-decoration: underline; color: #5a6cf8; }}
        .category a:visited {{ color: #8e9cfc; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #666; font-size: 14px; border-top: 1px solid #eee; }}
        .summary {{ background: #e8f5e8; border-left: 4px solid #28a745; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .summary h2 {{ color: #28a745; margin-top: 0; }}
        .risk-indicator {{ font-weight: bold; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }}
        .risk-high {{ background: #ffebee; color: #c62828; }}
        .risk-medium {{ background: #fff3e0; color: #ef6c00; }}
        .risk-low {{ background: #e8f5e8; color: #2e7d32; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Daily AI Newsletter Report</h1>
        <p>{date_str} • {total_items} AI items processed from {len(newsletters)} sources</p>
    </div>
"""
        
        # Add executive summary if provided
        if summary and total_items > 0:
            html += f"""
    <div class="summary">
        <h2>📊 Executive Summary</h2>
        <p>{summary}</p>
    </div>
"""
        
        if not categories or total_items == 0:
            html += '<div class="category"><h2>📭 No AI newsletters found</h2><p>No relevant AI content was found for today.</p></div>'
        else:
            for category, items in categories.items():
                if items:  # Only show categories with items
                    html += f"""
    <div class="category">
        <h2>{category}</h2>
        <ul>
"""
                    for item in items:
                        # Item already contains HTML links from fallback categorization
                        # Add risk indicator if available
                        risk_indicator = self._get_risk_indicator(item, risk_levels)
                        html += f"            <li>{item.strip()}{risk_indicator}</li>\n"
                    html += """        </ul>
    </div>"""
        
        html += """
    <div class="footer">
        <p>🤖 Generated by AI Newsletter Reporter • Powered by OpenAI</p>
    </div>
</body>
</html>
"""
        return html
    def _convert_urls_to_links(self,text:str)->str:
        m=re.match(r"(.+?)\s*\((https?://[^\s\)]+)\)",text)
        return f"{m.group(1)} <a href=\"{m.group(2)}\">[Link]</a>" if m else text
    
    def _get_risk_indicator(self, item: str, risk_levels: Dict[str, str]) -> str:
        """Get risk indicator HTML for an item"""
        if not risk_levels:
            return ""
        
        # Extract the item title (before " (Source:")
        item_title = item.split(" (Source:")[0].strip()
        # Remove HTML tags if present
        item_title = re.sub(r'<[^>]+>', '', item_title)
        
        # Find matching risk level
        risk_level = None
        for title, level in risk_levels.items():
            if title.lower() in item_title.lower() or item_title.lower() in title.lower():
                risk_level = level
                break
        
        if not risk_level:
            risk_level = "LOW"  # Default to LOW if not found
        
        risk_class = f"risk-{risk_level.lower()}"
        return f' <span class="risk-indicator {risk_class}">({risk_level})</span>'
    
    def _extract_url_from_body(self, body: str) -> str:
        """Extract URL from email body"""
        if not body:
            return ""
        
        # Look for URLs in the body text
        url_patterns = [
            r'https?://[^\s\)]+',  # Basic URL pattern
            r'\(https?://[^\s\)]+\)',  # URL in parentheses
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, body)
            if match:
                url = match.group(0).strip('()')
                return url
        
        return ""

def init_database(db_path:Path):
    conn=get_db_connection(db_path);cur=conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        headline TEXT NOT NULL,
        source TEXT NOT NULL,
        url TEXT,
        body_preview TEXT,
        date_processed TEXT NOT NULL,
        headline_sentiment REAL,
        body_sentiment REAL,
        retail_risk TEXT,
        systemic_risk_score INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_date_processed ON articles(date_processed)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON articles(source)")
    conn.commit();conn.close()

def main():
    setup_logging(debug=DEBUG)
    logger=logging.getLogger(__name__)
    logger.info("🚀 Starting AI Newsletter Reporter")
    config=Config();logger.debug(f"Config: max_emails={config.max_emails}, days_back={config.days_back}")
    init_database(config.db_path)
    gmail_srv=GmailService(config)
    email_filt=EmailFilter(config)
    newsapi_srv=NewsAPIService(config)
    rss_srv=RSSFeedService(config)
    llm_proc=LLMProcessor(config)
    
    # Fetch from multiple sources
    newsletters=gmail_srv.fetch_newsletters(email_filt)
    if newsapi_srv.enabled:
        newsletters+=newsapi_srv.fetch_ai_news()
    newsletters+=rss_srv.fetch_rss_content()
    if not newsletters:
        logger.warning("No content fetched; sending empty report.")
        html=llm_proc.format_report([], {})
        gmail_srv.send_email(html, "🤖 Daily AI Newsletter Report - No Content")
        return
    logger.info(f"Processing {len(newsletters)} items.")
    
    # Debug: Show sample content
    logger.info("📋 Sample collected items:")
    for i, nl in enumerate(newsletters[:5]):
        logger.info(f"  {i+1}. {nl['subject'][:80]}...")
    
    categories=llm_proc.categorize_newsletters(newsletters)
    logger.info(f"📂 Categories returned: {list(categories.keys())}")
    logger.info(f"📊 Total categorized items: {sum(len(items) for items in categories.values())}")
    
    summary=llm_proc.generate_summary(newsletters)
    risk_levels=llm_proc.assess_risk_levels(newsletters)
    logger.info(f"📊 Risk assessment completed: {len(risk_levels)} items analyzed")
    report_html=llm_proc.format_report(newsletters, categories, summary, risk_levels)
    subject=f"🤖 Daily AI Newsletter Report - {datetime.now().strftime('%Y-%m-%d')}"
    gmail_srv.send_email(report_html, subject)
    logger.info("✅ Report sent.")

if __name__=="__main__": main()
