import torch
import sys

# Ensure imports work regardless of how script is run
try:
    from .wikirace_env import WikiRaceEnv
    from .dqn_agent import DQNAgent
    from .shortest_path import bidirectional_bfs
except ImportError:
    from src.wikirace_env import WikiRaceEnv
    from src.dqn_agent import DQNAgent
    from src.shortest_path import bidirectional_bfs

MODEL_PATH = "dqn_wikirace.pt"
MAX_STEPS = 20

def run_custom_race(start_page: str, target_page: str):
    # 1. Initialize Environment
    print("Loading environment (this may take a few seconds)...")
    env = WikiRaceEnv(max_steps=MAX_STEPS)
    
    # 2. Validate Inputs
    if start_page not in env.valid_pages:
        print(f"Error: '{start_page}' is not in the 10,000-page dataset.")
        return
    if target_page not in env.valid_pages:
        print(f"Error: '{target_page}' is not in the 10,000-page dataset.")
        return

    # 3. Load Trained Agent
    print(f"Loading model from {MODEL_PATH}...")
    obs, info = env.reset(start=start_page, target=target_page)
    obs_dim = len(obs)
    
    # Initialize Dueling DQN Agent
    agent = DQNAgent(obs_dim=obs_dim, max_actions=32)
    try:
        agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
        agent.q_net.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file not found. Make sure dqn_wikirace.pt exists.")
        return

    # 4. Calculate Optimal Path (BFS) for comparison
    print(f"\nCalculating shortest path (BFS) baseline...")
    bfs_path = bidirectional_bfs(env.graph, start_page, target_page)
    bfs_len = len(bfs_path) - 1 if bfs_path else "Unreachable"
    print(f"Optimal BFS Distance: {bfs_len} clicks")

    # 5. Run the Race
    print(f"\n=== AI Agent Racing: '{start_page}' -> '{target_page}' ===\n")
    
    done = False
    path = [start_page]
    total_reward = 0.0

    while not done:
        # Get valid actions (sorted by relevance)
        valid_actions = env.available_actions()
        num_actions = len(valid_actions)
        
        # Agent decides
        action = agent.act(obs, min(num_actions, 32), epsilon=0.0)
        
        # Step
        next_obs, reward, done, info = env.step(action)
        
        # Logging
        current = info['current_page']
        path.append(current)
        total_reward += reward
        
        print(f"Step {info['step_count']}: Clicked -> {current}")
        
        obs = next_obs

    # 6. Result Summary
    print("\n------------------------------------------------")
    print(f"Race Finished!")
    print(f"Status: {info['reason']}")
    print(f"Agent Steps: {len(path) - 1}")
    print(f"Agent Path: {' -> '.join(path)}")
    print("------------------------------------------------")

if __name__ == "__main__":
    # You can change these to test different pairs
    START = "YouTube"
    TARGET = "India"
    
    # Allow command line args: python -m src.manual_race "Apple Inc." "Steve Jobs"
    if len(sys.argv) == 3:
        START = sys.argv[1]
        TARGET = sys.argv[2]
        
    run_custom_race(START, TARGET)