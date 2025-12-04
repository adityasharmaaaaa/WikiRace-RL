import numpy as np

try:
    from .wiki_graph import load_graph
    from .shortest_path import bidirectional_bfs
    from .config import EMBEDDINGS_FILE
except ImportError:
    from wiki_graph import load_graph
    from shortest_path import bidirectional_bfs
    EMBEDDINGS_FILE = "embeddings.npz"


def load_embeddings():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    pages = data["pages"]
    vectors = data["embeddings"]

    emb_dict = {}
    for title, vec in zip(pages, vectors):
        emb_dict[str(title)] = vec.astype(np.float32)

    return emb_dict


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def greedy_semantic_path(graph, embeddings, start: str, target: str, max_steps: int = 20):

    if start not in embeddings or target not in embeddings:
        print("Start or target missing embeddings. They may have no summary.")
        return None

    current = start
    path = [current]
    visited = {current}

    target_vec = embeddings[target]

    for _ in range(max_steps):
        neighbors = graph.get(current, [])
        best_neighbor = None
        best_score = -1.0

        for nb in neighbors:
            if nb in visited:
                continue
            if nb not in embeddings:
                continue

            score = cosine_sim(embeddings[nb], target_vec)

            if score > best_score:
                best_score = score
                best_neighbor = nb

        if best_neighbor is None:
            print("AI stuck — no neighbors with embeddings.")
            return path

        path.append(best_neighbor)
        visited.add(best_neighbor)  # <-- FIXED
        current = best_neighbor

        if current == target:
            return path

    return path


if __name__ == "__main__":
    print("Loading Graph and Embeddings...\n")

    graph = load_graph()
    embeddings = load_embeddings()

    start = "India"
    target = "YouTube"

    print(f"=== Shortest Path (Bidirectional BFS) from '{start}' to '{target}' ===")
    sp = bidirectional_bfs(graph, start, target)
    if sp:
        print(f"Length: {len(sp) - 1}")
        for p in sp:
            print(" ->", p)
    else:
        print("No BFS path found.")

    print(f"\n=== AI Greedy Semantic Path from '{start}' to '{target}' ===")
    gp = greedy_semantic_path(graph, embeddings, start, target, max_steps=15)
    if gp:
        print(f"Length: {len(gp) - 1}")
        for p in gp:
            print(" ->", p)
    else:
        print("No AI semantic path.")
