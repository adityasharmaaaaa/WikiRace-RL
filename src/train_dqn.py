# src/train_dqn.py

import numpy as np
import torch
import os
import random

try:
    from .wikirace_env import WikiRaceEnv
    from .dqn_agent import DQNAgent
except ImportError:
    from wikirace_env import WikiRaceEnv
    from dqn_agent import DQNAgent

# -------------------------
# Training Configuration
# -------------------------

NUM_EPISODES = 2000       
MAX_STEPS_PER_EPISODE = 20
BATCH_SIZE = 128
MAX_ACTIONS = 32           
TARGET_UPDATE_FREQ = 200   

# HER Probability: High because we need more signal
HER_PROBABILITY = 0.8     

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997     

MODEL_PATH = "dqn_wikirace.pt"
BEST_MODEL_PATH = "best_dqn_wikirace.pt"

def get_her_obs(env, current_page, target_page):
    """
    Helper to reconstruct an observation for a specific target
    without resetting the environment.
    """
    curr_vec = env.embeddings[current_page]
    tgt_vec = env.embeddings[target_page]
    return np.concatenate([curr_vec, tgt_vec], axis=0)

def train():
    env = WikiRaceEnv(max_steps=MAX_STEPS_PER_EPISODE)

    obs, info = env.reset()
    obs_dim = obs.shape[0]

    print(f"Observation dimension: {obs_dim}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize Agent (Now uses Dueling Network if you updated dqn_agent.py)
    agent = DQNAgent(
        obs_dim=obs_dim,
        max_actions=MAX_ACTIONS,
        lr=2e-4, 
        buffer_capacity=50000,
    )
    
    # Optional: Load previous weights if you want to continue training
    if os.path.exists(MODEL_PATH):
        try:
            agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
            agent.update_target()
            print("Loaded existing model weights.")
        except:
            print("Starting fresh model (incompatible weights or file not found).")

    epsilon = EPSILON_START
    reward_history = []
    success_count = 0
    
    # --- FIX: Initialize best_success_rate here ---
    best_success_rate = 0.0 
    # ----------------------------------------------

    print("\n=== Starting DQN Training (With Hindsight Experience Replay) ===\n")

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        # Temporary buffer to store this episode's path for HER
        episode_trajectory = []
        
        real_target = info["target"]
        current_page = info["start"]

        while not done:
            valid_actions = env.available_actions()
            num_actions = len(valid_actions)
            
            if num_actions == 0:
                break # Dead end

            num_actions_capped = min(num_actions, MAX_ACTIONS)
            action = agent.act(obs, num_actions_capped, epsilon)
            
            # Execute Step
            next_obs, reward, done, step_info = env.step(action)
            
            next_num_actions = len(env.available_actions())
            next_num_actions_capped = min(next_num_actions, MAX_ACTIONS)

            # 1. Store REAL transition
            agent.remember(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                next_num_actions=next_num_actions_capped,
            )

            # Save info for HER later
            episode_trajectory.append({
                "curr_page": current_page,
                "action": action,
                "next_page": step_info["current_page"],
                "next_num_actions": next_num_actions_capped,
                "done": done
            })

            obs = next_obs
            current_page = step_info["current_page"]
            episode_reward += reward
            
            # Training Step
            agent.train_step(BATCH_SIZE)
            if agent.train_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()
            
            if step_info["reason"] == "goal_reached":
                success_count += 1

        # --- Hindsight Experience Replay (HER) Block ---
        final_page = current_page
        
        # Only do HER if we actually moved and didn't reach the real target
        if final_page != real_target and len(episode_trajectory) > 0 and random.random() < HER_PROBABILITY:
            
            # Re-label the entire path with final_page as the goal
            for i, transition in enumerate(episode_trajectory):
                
                # Reconstruct Observation with FAKE target
                fake_obs = get_her_obs(env, transition["curr_page"], final_page)
                fake_next_obs = get_her_obs(env, transition["next_page"], final_page)
                
                # Reconstruct Reward
                is_fake_goal = (transition["next_page"] == final_page)
                
                if is_fake_goal:
                    fake_reward = env.goal_reward
                    fake_done = True
                else:
                    fake_reward = env.step_penalty
                    fake_done = False 

                # Store HER transition
                agent.remember(
                    state=fake_obs,
                    action=transition["action"],
                    reward=fake_reward,
                    next_state=fake_next_obs,
                    done=fake_done,
                    next_num_actions=transition["next_num_actions"]
                )
        # -----------------------------------------------

        reward_history.append(episode_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Logging & Checkpointing
        if episode % 50 == 0:
            avg_rew = np.mean(reward_history[-50:])
            sr = success_count / 50.0
            
            # --- Checkpoint Logic ---
            if sr >= best_success_rate and sr > 0.02: # Save if we see any decent progress (>2%)
                best_success_rate = sr
                torch.save(agent.q_net.state_dict(), BEST_MODEL_PATH)
                print(f"    >>> New Best Model Saved! (Success Rate: {sr:.2f})")
            # ------------------------
            
            success_count = 0
            print(
                f"Ep {episode} | "
                f"Avg Reward: {avg_rew:.2f} | "
                f"Success Rate: {sr:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    print("\n=== Training completed! Saving final model... ===")
    torch.save(agent.q_net.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()