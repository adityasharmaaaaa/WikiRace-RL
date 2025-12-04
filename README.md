# WikiRace-RL# 🧠 WikiRace AI: Autonomous Semantic Navigation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Reinforcement Learning](https://img.shields.io/badge/Skill-Reinforcement%20Learning-red)

An autonomous AI agent that learns to play the "Wiki Game" (navigating from one Wikipedia page to another) using Deep Reinforcement Learning.

Unlike standard graph traversal algorithms, this agent uses **Semantic Understanding** (BERT embeddings) to navigate. It knows that "DNA" is related to "Biology" and "Apple" is related to "Technology," allowing it to traverse a graph of **10,000+ nodes** without a map.

## 🚀 Key Features

* **Dueling DQN Architecture:** Splits Value and Advantage streams for stable learning.
* **Hindsight Experience Replay (HER):** Learns from failed paths, boosting success rate from 0% to 73%.
* **Semantic Navigation:** Uses `sentence-transformers` to "read" page content.
* **Custom Environment:** OpenAI Gym-compliant environment with history masking to prevent loops.

## 📊 Performance

| Metric | Baseline (Random) | Initial DQN | **Dueling DQN + HER (Final)** |
| :--- | :---: | :---: | :---: |
| **Success Rate** | ~0% | 2% | **73%** |

## 🛠️ Usage

**Run a manual race:**
```bash
python -m src.manual_race "YouTube" "India"
