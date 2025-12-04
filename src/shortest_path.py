from typing import Dict, List, Optional

# Support both module and direct execution
try:
    from .wiki_graph import load_graph
except ImportError:
    from wiki_graph import load_graph


def bidirectional_bfs(graph: Dict[str, List[str]], start: str, target: str) -> Optional[List[str]]:
    if start not in graph or target not in graph:
        return None
    if start == target:
        return [start]

    # BFS frontiers
    front_start = {start}
    front_target = {target}

    # Parent maps
    parents_start = {start: None}
    parents_target = {target: None}

    while front_start and front_target:

        # ---- Expand from start side ----
        next_front = set()
        for node in front_start:
            for neighbor in graph.get(node, []):
                if neighbor not in parents_start:
                    parents_start[neighbor] = node
                    next_front.add(neighbor)

        front_start = next_front

        meet_point = front_start & front_target
        if meet_point:
            meet = meet_point.pop()
            return reconstruct_path(meet, parents_start, parents_target)

        # ---- Expand from target side (reverse scan) ----
        next_front = set()
        for node in front_target:
            for src, neighbors in graph.items():
                if node in neighbors and src not in parents_target:
                    parents_target[src] = node
                    next_front.add(src)

        front_target = next_front

        meet_point = front_start & front_target
        if meet_point:
            meet = meet_point.pop()
            return reconstruct_path(meet, parents_start, parents_target)

    return None


def reconstruct_path(meet: str, parents_start: Dict[str, str], parents_target: Dict[str, str]) -> List[str]:
    # Path from start → meet
    path_start = []
    node = meet
    while node is not None:
        path_start.append(node)
        node = parents_start[node]
    path_start.reverse()

    # Path from meet → target
    path_target = []
    node = parents_target[meet]
    while node is not None:
        path_target.append(node)
        node = parents_target[node]

    return path_start + path_target


if __name__ == "__main__":
    graph = load_graph()

    start = "Animal (2023 film)"
    target = "India"

    print(f"Finding shortest path from '{start}' to '{target}'...")

    path = bidirectional_bfs(graph, start, target)

    if path:
        print(f"\nShortest path ({len(path)-1} clicks):")
        for p in path:
            print(" ->", p)
    else:
        print("No path found.")


