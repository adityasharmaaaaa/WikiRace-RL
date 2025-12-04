import json
from collections import deque
from typing import Dict, List, Set
import wikipediaapi
from tqdm import tqdm
import requests
import time

try:
    from .config import PAGES_FILE, GRAPH_FILE, MAX_PAGES, MAX_LINKS_PER_PAGE
except ImportError:
    PAGES_FILE = "pages.json"
    GRAPH_FILE = "graph.json"
    MAX_PAGES = 10000
    MAX_LINKS_PER_PAGE = 150

def get_most_viewed_pages(limit=50):
    # Expanded seed list to cover more topics immediately
    return [
        "United States", "China", "India", "United Kingdom", "World War II",
        "Artificial intelligence", "Python (programming language)", "Earth",
        "Google", "Elon Musk", "Facebook", "YouTube", "Apple Inc.",
        "Biology", "Physics", "Chemistry", "Mathematics", "History",
        "Donald Trump", "Barack Obama", "Cristiano Ronaldo", "Lionel Messi",
        "Taylor Swift", "Michael Jackson", "Star Wars", "Harry Potter"
    ]

def get_wiki_api():
    return wikipediaapi.Wikipedia(
        user_agent="WikiRaceBot/2.0 (http://example.com/bot; email@example.com)",
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

def fetch_links(api, title: str) -> List[str]:
    try:
        page = api.page(title)
        if not page.exists():
            return []
        return list(page.links.keys())[:MAX_LINKS_PER_PAGE]
    except Exception:
        return []

def build_graph(seed_titles: List[str]) -> Dict[str, List[str]]:
    api = get_wiki_api()
    graph: Dict[str, List[str]] = {}
    visited: Set[str] = set()
    queue = deque(seed_titles)
    
    # Keep track of pages we encountered but haven't visited yet
    # to prioritize them if queue runs dry
    seen_links = set(seed_titles)

    pbar = tqdm(total=MAX_PAGES, desc="Building Graph")

    while len(graph) < MAX_PAGES and queue:
        title = queue.popleft()
        
        if title in graph:
            continue

        try:
            links = fetch_links(api, title)
            # Filter junk immediately
            clean_links = [
                l for l in links 
                if "Identifier" not in l and "ISBN" not in l and ":" not in l
            ]
            
            graph[title] = clean_links
            pbar.update(1)

            # Add new links to queue
            for link in clean_links:
                if link not in seen_links:
                    seen_links.add(link)
                    queue.append(link)
        
        except Exception as e:
            print(f"Error processing {title}: {e}")

    pbar.close()
    return graph

def save_graph(graph: Dict[str, List[str]]):
    pages = list(graph.keys())
    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f)

def load_graph():
    """
    Load the saved Wikipedia graph from disk.
    Used by wikirace_env.py
    """
    import os
    if not os.path.exists(GRAPH_FILE):
        print(f"Graph file not found at {GRAPH_FILE}. Please run wiki_graph.py first.")
        return {}
        
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
    
if __name__ == "__main__":
    seeds = get_most_viewed_pages()
    print(f"Starting crawl with {len(seeds)} seed pages...")
    graph = build_graph(seeds)
    save_graph(graph)
    print(f"Done! Saved {len(graph)} pages.")