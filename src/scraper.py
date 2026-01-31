import argparse
import requests
from bs4 import BeautifulSoup
import os


def scrape_wikipedia(topic: str, out_path: str = "data/article.txt"):
    """Find the closest Wikipedia article and save its text to a .txt file."""
    # Use a session with a browser-like User-Agent to reduce request blocking.
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Prefer the MediaWiki 'query' API for stable search results.
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}

    try:
        resp = session.get(api_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            title = search_results[0].get("title")
            article_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        else:
            # No results from API; fall back to the web search page.
            raise RuntimeError("No results from MediaWiki API")
    except Exception:
        try:
            search_page = session.get("https://en.wikipedia.org/w/index.php", params={"search": topic}, headers=headers, allow_redirects=True, timeout=10)
            search_page.raise_for_status()
            article_url = search_page.url
            if "search=" in article_url or "Special:Search" in article_url:
                print("No Wikipedia article found via fallback search page.")
                return None
        except Exception as e:
            print(f"Failed to locate article via API and fallback: {e}")
            return None

    page = session.get(article_url, headers=headers, timeout=10)
    try:
        page.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error when fetching article page: {e}")
        return None
    soup = BeautifulSoup(page.content, "html.parser")
    text = "\n".join([p.get_text() for p in soup.find_all("p")])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully scraped: {article_url}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Topic to search")
    parser.add_argument("--out", default="data/article.txt", help="Output text file")
    args = parser.parse_args()
    scrape_wikipedia(args.topic, args.out)
