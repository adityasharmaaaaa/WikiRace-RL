import json
import requests
import numpy as np
from tqdm import tqdm

try:
    from .config import PAGES_FILE, EMBEDDINGS_FILE
except ImportError:
    PAGES_FILE = "pages.json"
    EMBEDDINGS_FILE = "embeddings.npz"

from sentence_transformers import SentenceTransformer

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def load_pages():
    with open(PAGES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_summary(title: str) -> str:
    url = WIKI_SUMMARY_URL + requests.utils.quote(title)

    headers = {
        "User-Agent": "WikiRaceBot/1.0 (http://example.com/bot; email@example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "") or ""
    except Exception as e:
        print(f"Error fetching summary for {title}: {e}")

    return ""


def build_embeddings():
    pages = load_pages()
    print(f"Loaded {len(pages)} pages from {PAGES_FILE}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    summaries = []
    valid_pages = []

    print("\nFetching summaries from Wikipedia...")
    for title in tqdm(pages):
        text = fetch_summary(title)
        if text.strip():
            summaries.append(text)
            valid_pages.append(title)

    print(f"\nEncoding {len(valid_pages)} pages into embeddings...\n")

    vectors = model.encode(
        summaries,
        show_progress_bar=True,  # <-- FIXED
        convert_to_numpy=True
    )

    np.savez_compressed(
        EMBEDDINGS_FILE,
        pages=np.array(valid_pages),
        embeddings=vectors
    )

    print(f"\nSaved embeddings to {EMBEDDINGS_FILE}")
    print(f"Total embeddings: {len(valid_pages)} pages")


if __name__ == "__main__":
    build_embeddings()
