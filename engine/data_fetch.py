import yfinance as yf

def fetch_nifty_news(ticker_symbol):
    """
    Fetches the latest news headlines for a specific Nifty 50 stock.
    Enhanced to handle nested structures in the 2026 yfinance API.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        news = stock.news
        
        if not news:
            return ["No recent news found for this ticker."]

        headlines = []
        for item in news[:5]:
            # Professional 'Deep Get' strategy:
            # Check for 'title', then 'headline', then look inside 'content' if it exists
            title = item.get('title') or item.get('headline')
            
            # If still not found, some versions nest it under 'content' -> 'title'
            if not title and 'content' in item:
                title = item['content'].get('title') or item['content'].get('headline')
            
            # Final fallback
            headlines.append(title if title else "Market update available (click to view)")
            
        return headlines
    except Exception as e:
        return [f"Connection error: {str(e)}"]