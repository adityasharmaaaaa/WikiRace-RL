import numpy as np
import torch

try:
    from .wikirace_env import WikiRaceEnv
    from .dqn_agent import DQNAgent
    from .shortest_path import bidirectional_bfs
except ImportError:
    from wikirace_env import WikiRaceEnv
    from dqn_agent import DQNAgent
    from shortest_path import bidirectional_bfs


MODEL_PATH = "dqn_wikirace.pt"
N_EVAL_EPISODES = 30


def evaluate_agent(n_episodes=N_EVAL_EPISODES):

    env = WikiRaceEnv(max_steps=32)


    # ---- FIX: reset first, THEN compute obs_dim ----
    obs, info = env.reset()
    obs_dim = len(obs)
    max_actions = env.max_steps

    print(f"Observation dim: {obs_dim}")
    print(f"Max actions: {max_actions}")

    # Load trained DQN
    agent = DQNAgent(obs_dim=obs_dim, max_actions=max_actions)
    agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
    agent.q_net.eval()

    print("\n=== Loaded trained DQN model ===\n")

    successes = 0
    total_steps = []
    examples = []

    # ---- Evaluation Loop ----
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()

        start = info["start"]
        target = info["target"]
        bfs_len = info["optimal_length"]

        path = [start]
        done = False
        step_count = 0

        while not done:
            num_actions = len(env.available_actions())
            action = agent.act(obs, num_actions=num_actions, epsilon=0.0)

            next_obs, reward, done, step_info = env.step(action)
            obs = next_obs
            step_count += 1
            path.append(step_info["current_page"])

            if done:
                if step_info["reason"] == "goal_reached":
                    successes += 1

                total_steps.append(step_count)

                examples.append({
                    "start": start,
                    "target": target,
                    "bfs_length": bfs_len,
                    "agent_length": step_count,
                    "agent_path": path,
                    "status": step_info["reason"],
                })
                break

        print(f"Episode {ep}/{n_episodes} | "
              f"Status: {examples[-1]['status']} | "
              f"Steps: {step_count} | BFS: {bfs_len}")

    # ---- Summary ----
    print("\n===============================")
    print("       EVALUATION SUMMARY")
    print("===============================\n")

    print(f"Total episodes: {n_episodes}")
    print(f"Successes: {successes}")
    print(f"Success rate: {successes/n_episodes:.2f}")

    if total_steps:
        print(f"Avg steps (all episodes): {np.mean(total_steps):.2f}")

    print("\n--- Example Episodes ---")
    for ex in examples[:5]:
        print("\n----------------------------------")
        print(f"Start: {ex['start']}")
        print(f"Target: {ex['target']}")
        print(f"BFS Length: {ex['bfs_length']}")
        print(f"Agent Length: {ex['agent_length']}")
        print("Agent Path:")
        for p in ex["agent_path"]:
            print(" ->", p)
        print(f"Status: {ex['status']}")
        print("----------------------------------")


if __name__ == "__main__":
    evaluate_agent()
