import random
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from .wiki_graph import load_graph
    from .shortest_path import bidirectional_bfs
    from .config import EMBEDDINGS_FILE
except ImportError:
    from wiki_graph import load_graph
    from shortest_path import bidirectional_bfs
    EMBEDDINGS_FILE = "embeddings.npz"

def load_embeddings() -> Dict[str, np.ndarray]:
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    pages = data["pages"]
    vectors = data["embeddings"]
    emb_dict = {}
    for title, vec in zip(pages, vectors):
        emb_dict[str(title)] = vec.astype(np.float32)
    return emb_dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class WikiRaceEnv:
    def __init__(
        self,
        graph: Optional[Dict[str, List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        max_steps: int = 20,
        goal_reward: float = 10.0,
        step_penalty: float = -0.1,
        dead_end_penalty: float = -1.0,
        max_step_penalty: float = -1.0,
    ):
        self.graph = graph if graph is not None else load_graph()
        self.embeddings = embeddings if embeddings is not None else load_embeddings()
        
        self.valid_pages = [p for p in self.graph.keys() if p in self.embeddings]
        
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.dead_end_penalty = dead_end_penalty
        self.max_step_penalty = max_step_penalty
        
        self.current_page: Optional[str] = None
        self.target_page: Optional[str] = None
        self.start_page: Optional[str] = None
        
        self.visited: set = set()
        self.step_count: int = 0
        self.current_neighbors: List[str] = []
        self.optimal_length: Optional[int] = None

    def _get_obs(self) -> np.ndarray:
        curr_vec = self.embeddings[self.current_page]
        tgt_vec = self.embeddings[self.target_page]
        return np.concatenate([curr_vec, tgt_vec], axis=0)

    def _sort_neighbors_by_relevance(self, neighbors: List[str]) -> List[str]:
        if not self.target_page:
            return neighbors  
        target_vec = self.embeddings[self.target_page]
        scored_neighbors = []
        for nb in neighbors:
            nb_vec = self.embeddings[nb]
            score = cosine_similarity(nb_vec, target_vec)
            scored_neighbors.append((score, nb))
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        return [nb for _, nb in scored_neighbors]

    def _get_valid_neighbors(self, page: str) -> List[str]:
        raw_neighbors = self.graph.get(page, [])
        valid = []
        
        # --- FIX: Block Junk Pages ---
        blacklist_terms = ["(identifier)", "ISBN", "ISSN", "PMID", "Doi", "S2CID", "Ars Technica", "Wayback Machine"]
        
        for nb in raw_neighbors:
            # 1. Must have embedding
            if nb not in self.embeddings: continue
            
            # 2. Must NOT be visited
            if nb in self.visited: continue
            
            # 3. Must NOT be a junk page (The Identifier Trap)
            is_junk = False
            for term in blacklist_terms:
                if term in nb:
                    is_junk = True
                    break
            if is_junk: continue

            valid.append(nb)
            
        return self._sort_neighbors_by_relevance(valid)

    def reset(
            self,
            start: Optional[str] = None,
            target: Optional[str] = None,
            ensure_reachable: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        self.visited.clear()
        self.step_count = 0
        
        # --- NEW LOGIC: CURRICULUM LEARNING ---
        # With 50% probability, force a target that is close (1-3 steps away)
        # This helps the agent learn basic semantic associations first.
        force_close_target = (random.random() < 0.5)
        # --------------------------------------

        if start and target:
            self.start_page = start
            self.target_page = target
            self.current_page = start
            path = bidirectional_bfs(self.graph, start, target) if ensure_reachable else None
            self.optimal_length = len(path) - 1 if path else 0
        else:
            while True:
                s = random.choice(self.valid_pages)
                t = random.choice(self.valid_pages)
                if s == t: continue
                
                path = bidirectional_bfs(self.graph, s, t)
                if ensure_reachable:
                    if path is None:
                        continue
                    
                    dist = len(path) - 1
                    
                    # If we want easy mode, retry if the target is too far
                    if force_close_target and dist > 3:
                        continue
                        
                    self.optimal_length = dist
                else:
                    self.optimal_length = None
                
                self.start_page = s
                self.target_page = t
                self.current_page = s
                break
        
        self.visited.add(self.current_page)
        self.current_neighbors = self._get_valid_neighbors(self.current_page)
        
        return self._get_obs(), {
            "start": self.start_page,
            "target": self.target_page,
            "optimal_length": self.optimal_length,
            "neighbors": self.current_neighbors,
        }

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_dist = cosine_similarity(
            self.embeddings[self.current_page], 
            self.embeddings[self.target_page]
        )

        if action_index < 0 or action_index >= len(self.current_neighbors):
            return self._get_obs(), self.dead_end_penalty, True, {
                "reason": "invalid_action",
                "current_page": self.current_page
            }

        next_page = self.current_neighbors[action_index]
        self.current_page = next_page
        self.visited.add(next_page)
        self.step_count += 1
        
        done = False
        reward = self.step_penalty
        reason = "step"

        if self.current_page == self.target_page:
            reward = self.goal_reward
            done = True
            reason = "goal_reached"
        else:
            curr_dist = cosine_similarity(
                self.embeddings[self.current_page], 
                self.embeddings[self.target_page]
            )
            diff = curr_dist - prev_dist
            reward += (diff * 2.0) # Strengthen semantic signal

        self.current_neighbors = self._get_valid_neighbors(self.current_page)

        if not done:
            if not self.current_neighbors:
                reward += self.dead_end_penalty
                done = True
                reason = "dead_end"
            elif self.step_count >= self.max_steps:
                reward += self.max_step_penalty
                done = True
                reason = "max_steps"

        info = {
            "reason": reason,
            "current_page": self.current_page,
            "step_count": self.step_count,
            "neighbors": self.current_neighbors,
            "target": self.target_page
        }

        return self._get_obs(), reward, done, info

    def available_actions(self) -> List[int]:
        return list(range(len(self.current_neighbors)))
    
if __name__=="__main__":

    env=WikiRaceEnv(max_steps=20)

    obs,info=env.reset()
    print("Start:",info["start"])
    print("Target:",info["target"])
    print("Optimal BFS Length:",info["optimal_length"])
    print("Initial neighbors:",info["neighbors"])

    done=False
    total_reward=0.0

    while not done:
        actions=env.available_actions()
        if not actions:
            print("No available actions - stopping.")
            break

        a=random.choice(actions)
        next_obs,reward,done,step_info=env.step(a)
        total_reward+=reward

        print(
            f"Step {step_info['step_count']}: "
            f"Clicked -> {step_info['current_page']} | "
            f"Reason: {step_info['reason']} | Reward: {reward}"
        )
    print("Episode finished. Total reward:",total_reward)


