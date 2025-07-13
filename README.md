# 🤖 AI Newsletter Reporter - Enhanced Edition

A comprehensive AI news aggregation system that collects, filters, and delivers high-quality AI content from multiple sources into a beautifully formatted newsletter.

## 🚀 Features

### 📧 **Multi-Source Content Aggregation**
- **Gmail Integration**: Automatically processes labeled AI newsletters and performs intelligent content search
- **NewsAPI Integration**: Fetches from premium sources (WSJ, Bloomberg, Reuters, Financial Times) + comprehensive AI coverage
- **RSS Feed Integration**: Monitors 9+ major AI blogs and publications
- **Premium Source Priority**: Enhanced filtering for trusted media sources

### 🎯 **Intelligent Content Filtering**
- **AI Relevance Scoring**: Advanced algorithm to ensure high-quality AI-focused content
- **Premium Source Recognition**: Lower thresholds for established publications
- **Financial Content Filtering**: Excludes irrelevant stock/trading news while preserving AI business coverage
- **Comprehensive Deduplication**: Removes duplicate content across all sources

### 📊 **Professional Newsletter Generation**
- **Executive Summary**: AI-generated overview of key developments
- **Smart Categorization**: Organizes content into relevant categories
- **Beautiful HTML Formatting**: Professional styling with responsive design
- **Source Attribution**: Clear labeling of premium vs. regular sources

## 🔧 Installation & Setup

### 1. Prerequisites
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install openai requests beautifulsoup4 python-dotenv google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client tenacity feedparser
```

### 2. Environment Configuration
Create a `.env` file with the following variables:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
NEWSAPI_KEY=your_newsapi_key_here

# Gmail Configuration
GMAIL_CREDENTIALS_PATH=path/to/your/credentials.json
TO_EMAIL=recipient@email.com
FROM_EMAIL=sender@email.com

# Processing Configuration (Optional - defaults shown)
MAX_EMAILS=50
DAYS_BACK=7
DEBUG=false

# Gmail Label (Optional)
GMAIL_LABEL=ai-newsletters

# Content Filtering (Optional)
NEWSLETTER_KEYWORDS=AI,artificial intelligence,machine learning,newsletter,digest,weekly,update
EXCLUDED_KEYWORDS=unsubscribe,spam,promotion,offer,sale,discount
```

### 3. Gmail API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Gmail API
3. Create OAuth 2.0 credentials
4. Download `credentials.json` file to your project directory

### 4. Newsletter Subscriptions
For maximum content quality, subscribe to these newsletters and label them as "AI Newsletter" in Gmail:

**Premium Publications:**
- **WSJ**: "The Future of Everything" newsletter
- **Bloomberg**: "Fully Charged" (tech newsletter)
- **Financial Times**: "#techFT" newsletter
- **Reuters**: "Tech" newsletter

**AI-Focused Sources:**
- OpenAI updates
- Anthropic newsletters
- Google AI updates
- Microsoft AI newsletters

## 🎯 Content Sources

### 📧 **Gmail Sources**
- **Labeled Newsletters**: Emails tagged with "AI Newsletter" label
- **Premium Media Search**: Direct searches for WSJ, Bloomberg, Reuters, FT, NYT
- **Tech Publication Search**: TechCrunch, The Verge, Wired, Ars Technica
- **AI Company Search**: OpenAI, Anthropic, DeepMind, Microsoft, Google
- **Comprehensive AI Search**: 10+ different search strategies

### 📰 **NewsAPI Sources**
- **Premium Domains**: WSJ, Bloomberg, Reuters, Financial Times, NYT, Washington Post
- **Tech Publications**: TechCrunch, The Verge, Wired, Ars Technica, VentureBeat, Engadget
- **Comprehensive AI Search**: Multiple queries covering different AI aspects
- **Emerging Topics**: AI safety, ethics, regulation, governance

### 📡 **RSS Feed Sources**
- Google AI Blog
- OpenAI Blog  
- Microsoft AI Blog
- Meta Research Blog
- VentureBeat AI
- TechCrunch AI
- The Verge AI
- Wired AI
- O'Reilly Radar

## 🏗️ Architecture

### Core Components

1. **EmailFilter**: Intelligent content filtering with AI relevance scoring
2. **GmailService**: Comprehensive Gmail integration with multiple search strategies
3. **NewsAPIService**: Enhanced API integration with premium source targeting
4. **RSSFeedService**: RSS feed monitoring and processing
5. **LLMProcessor**: AI-powered categorization and summary generation

### Processing Pipeline

1. **Content Collection**: Fetch from Gmail, NewsAPI, and RSS feeds
2. **Filtering & Scoring**: Apply AI relevance scoring and exclude irrelevant content
3. **Deduplication**: Remove duplicate content across all sources
4. **AI Processing**: Generate executive summary and categorize content
5. **Report Generation**: Create professional HTML newsletter
6. **Delivery**: Send via Gmail API

## 🎨 Newsletter Format

### Structure
- **Header**: Date, source count, and item count
- **Executive Summary**: AI-generated overview of key developments
- **Categorized Content**:
  - 🏛️ AI Regulation & Policy
  - 🔧 AI Tools & Products
  - 🔬 AI Research & Development
  - 🏢 AI Business & Industry
  - ⚖️ AI Ethics & Safety
  - 📰 Other AI News

### Features
- Professional HTML styling with gradients and typography
- Responsive design for mobile and desktop
- Clickable links with proper formatting
- Source attribution with premium indicators
- Clean, readable layout

## 🔍 Content Quality Features

### Advanced Filtering
- **AI Relevance Scoring**: Ensures high-quality, AI-focused content
- **Premium Source Bonus**: Trusted sources get priority
- **Financial Content Filtering**: Excludes irrelevant stock news
- **Comprehensive Search**: 20+ different search strategies

### Deduplication
- **Cross-Source**: Removes duplicates between Gmail, NewsAPI, and RSS
- **Title-Based**: Prevents similar articles from multiple sources
- **URL-Based**: Ensures unique content delivery

### Source Quality
- **Premium Recognition**: WSJ, Bloomberg, Reuters, FT, NYT prioritized
- **Tech Publication Coverage**: TechCrunch, Wired, The Verge, etc.
- **AI Company Direct**: OpenAI, Anthropic, Google, Microsoft blogs
- **Research Sources**: Academic and research publication coverage

## 🚀 Usage

### Basic Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run the newsletter generator
python3 ai_newsletter_reporter.py
```

### Advanced Configuration
```bash
# Enable debug mode for troubleshooting
DEBUG=1 python3 ai_newsletter_reporter.py

# Increase content volume
MAX_EMAILS=75 DAYS_BACK=10 python3 ai_newsletter_reporter.py
```

## 📊 Expected Output

### Content Volume
- **60-100+ AI items** per newsletter
- **Multiple premium sources** (WSJ, Bloomberg, Reuters, etc.)
- **Comprehensive coverage** across all AI domains
- **High relevance** with minimal noise

### Source Distribution
- **Gmail**: 20-50 items from newsletters and searches
- **NewsAPI**: 30-75 items from premium and tech sources  
- **RSS Feeds**: 15-45 items from AI blogs and research

## 🔧 Troubleshooting

### Common Issues

1. **Gmail Label Not Found**
   - Check that "AI Newsletter" label exists in Gmail
   - Verify exact spelling and capitalization
   - Check debug logs for available labels

2. **Low Content Volume**
   - Increase `MAX_EMAILS` in .env file
   - Subscribe to more AI newsletters
   - Check NewsAPI quota and key validity

3. **Authentication Errors**
   - Verify Gmail credentials.json is valid
   - Check OAuth scopes in Google Cloud Console
   - Re-run OAuth flow if token.json is corrupted

4. **Missing Dependencies**
   ```bash
   pip install feedparser  # For RSS functionality
   ```

### Debug Mode
Enable detailed logging:
```bash
DEBUG=1 python3 ai_newsletter_reporter.py
```

## 📈 Performance Metrics

### Content Processing
- **Sources Monitored**: 50+ different sources
- **Search Strategies**: 20+ different Gmail searches
- **API Calls**: 4+ NewsAPI queries per run
- **RSS Feeds**: 9+ AI-focused feeds

### Quality Assurance
- **AI Relevance Filtering**: Multi-layer content validation
- **Premium Source Priority**: Enhanced coverage from trusted sources
- **Deduplication**: Cross-source duplicate removal
- **Executive Summary**: AI-generated key insights

## 🔄 Automation

### Cron Job Setup
For daily newsletters, add to crontab:
```bash
# Daily at 8 AM
0 8 * * * cd /path/to/newsletter && source venv/bin/activate && python3 ai_newsletter_reporter.py
```

### Error Handling
- Automatic fallback processing
- Email delivery on failures
- Comprehensive logging
- Graceful degradation when sources are unavailable

## 📝 Changelog

### v2.0 - Enhanced Content Retrieval
- ✅ **Multi-source integration**: NewsAPI + RSS feeds + Gmail
- ✅ **Premium source targeting**: WSJ, Bloomberg, Reuters, FT
- ✅ **Enhanced Gmail search**: 10+ search strategies
- ✅ **RSS feed integration**: 9+ AI-focused feeds
- ✅ **Increased content limits**: 50-100+ items per newsletter
- ✅ **Executive summary**: AI-generated overview
- ✅ **Professional formatting**: Enhanced HTML styling
- ✅ **Advanced filtering**: AI relevance scoring
- ✅ **Cross-source deduplication**: Unique content guarantee

### v1.0 - Initial Release
- Basic Gmail newsletter processing
- Simple content filtering
- Basic HTML formatting

## 📄 License

This project is for personal and educational use. Please respect API terms of service and content licensing when using with commercial applications.

## 🤝 Contributing

To improve the newsletter:
1. Add new RSS feeds to `RSSFeedService`
2. Enhance AI relevance scoring in `EmailFilter`
3. Improve categorization prompts in `LLMProcessor`
4. Add new search strategies to Gmail content analysis

---

**Powered by OpenAI GPT for content processing and categorization** 🤖