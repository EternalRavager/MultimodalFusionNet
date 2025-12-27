import tldextract
import re

def extract_url_features(url):
    """
    Extracts structural and lexical features from a URL.
    
    Why we do this: Neural networks handle raw text well, but URLs have specific 
    structures (like TLDs, subdomains, and special chars) that contain strong 
    signals about the website's nature (e.g., many hyphens might indicate SEO spam).
    """
    ext = tldextract.extract(url)
    
    # --- Structural Features ---
    url_length = len(url)
    dot_count = url.count('.')
    dash_count = url.count('-')
    slash_count = url.count('/')
    digit_count = sum(c.isdigit() for c in url)
    
    # Check for HTTPS (secure sites are less likely to be spam/malicious, though not guaranteed)
    is_https = 1 if url.startswith('https') else 0
    tld_length = len(ext.suffix)
    
    # --- Advanced Features ---
    underscore_count = url.count('_')
    query_params = 1 if '?' in url else 0
    
    # Subdomains often indicate organizational structure (e.g., blog.site.com)
    subdomain_count = len(ext.subdomain.split('.')) if ext.subdomain else 0
    
    # Calculate how deep the URL goes (e.g., .com/folder/subfolder/file)
    path_depth = url.count('/') - 2 if url.startswith('http') else url.count('/')
    
    # Check for non-standard ports (common in dev or suspicious sites)
    has_port = 1 if ('//' in url and ':' in url.split('//')[1].split('/')[0]) else 0
    domain_length = len(ext.domain)
    
    # --- Keyword Heuristics ---
    # While the LSTM handles semantic meaning, explicit keyword flags give the 
    # model a "head start" on obvious categories.
    url_lower = url.lower()
    category_keywords = {
        'news': ['news', 'article', 'breaking', 'press', 'media', 'journal', 'times', 'post', 'gazette'],
        'shop': ['shop', 'store', 'buy', 'cart', 'product', 'sale', 'marketplace', 'ecommerce'],
        'sports': ['sport', 'game', 'team', 'league', 'soccer', 'football', 'basketball', 'nfl', 'nba'],
        'health': ['health', 'medical', 'doctor', 'hospital', 'wellness', 'fitness', 'clinic'],
        'tech': ['tech', 'software', 'app', 'digital', 'computer', 'android', 'ios', 'code'],
        'edu': ['edu', 'school', 'university', 'college', 'learn', 'course', 'academic'],
        'gov': ['gov', 'government', 'official', 'public', 'state', 'federal'],
        'blog': ['blog', 'post', 'article', 'diary', 'journal'],
        'social': ['social', 'community', 'forum', 'chat', 'connect'],
        'video': ['video', 'watch', 'tube', 'stream', 'movie', 'film'],
        'music': ['music', 'song', 'audio', 'listen', 'sound', 'radio'],
        'game': ['game', 'play', 'gaming', 'gamer'],
        'adult': ['adult', 'xxx', 'porn', 'sex'],
        'kids': ['kid', 'child', 'toy', 'cartoon', 'family'],
    }
    
    keyword_matches = sum(1 for keywords in category_keywords.values() 
                         for keyword in keywords if keyword in url_lower)
    
    # Return as a list of numerical features
    features = [
        url_length, dot_count, dash_count, slash_count, digit_count,
        is_https, tld_length, underscore_count, query_params,
        subdomain_count, path_depth, has_port, domain_length,
        keyword_matches
    ]
    return features

def clean_text(text):
    """
    Basic text preprocessing to normalize inputs for the LSTM.
    Converts to lowercase and removes special characters to reduce noise.
    """
    text = str(text).lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text