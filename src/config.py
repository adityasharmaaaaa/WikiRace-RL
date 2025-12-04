import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,"data")

os.makedirs(DATA_DIR,exist_ok=True)

PAGES_FILE=os.path.join(DATA_DIR,"pages.json")
GRAPH_FILE=os.path.join(DATA_DIR,"graph.json")
EMBEDDINGS_FILE=os.path.join(DATA_DIR,"embeddings.npz")

MAX_PAGES=10000
MAX_LINKS_PER_PAGE=200