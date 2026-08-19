# Deep Reinforcement Learning for Montezuma's Revenge: Curiosity-Driven Exploration & Dueling DQN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Atari-0298c3.svg)](https://gymnasium.farama.org/)
[![ALE](https://img.shields.io/badge/ALE-MontezumaRevenge--v5-green.svg)](https://ale.farama.org/)

This repository provides an implementation of Deep Reinforcement Learning (DRL) algorithms to tackle **Montezuma's Revenge** (`ALE/MontezumaRevenge-v5`), one of the hardest sparse-reward exploration benchmarks in the Atari 2600 suite. 

The project evaluates and compares:
1. **Actor-Critic (A2C) with Recurrent GRU and Generalized Advantage Estimation (GAE)** combined with an **Intrinsic Curiosity Module (ICM)** (Forward & Inverse Dynamics).
2. **Dueling Deep Q-Network (Dueling DQN)** with **Prioritized Experience Replay (PER)**.

---


## 🧠 Algorithms & Architectural Details

### 1. Actor-Critic with Recurrent Core (GRU) & GAE (`actor_critic.py`)
* **Visual Feature Extraction:** 4 Convolutional layers (`Conv2d` + `ELU`) processing 4-stacked grayscale Atari frames ($84 \times 84$).
* **Recurrent Memory:** 256-unit `GRUCell` for temporal state tracking.
* **Policy & Value Heads:** Separate linear layers producing action probabilities $\pi(a|s)$ and state-value estimates $V(s)$.
* **Advantage Estimation:** Generalized Advantage Estimation (GAE with $\gamma=0.99, \tau=1.0$) with entropy regularization loss.

### 2. Intrinsic Curiosity Module - ICM (`curiosity.py` & `curiosity_inverse.py`)
* **Feature Encoder:** CNN embedding state representations into a 512-dimensional latent feature space $\phi(s)$.
* **Forward Dynamics Model:** Predicts next-state features $\hat{\phi}(s_{t+1})$ given $\phi(s_t)$ and action $a_t$. Prediction error ($L_1 + L_2$ loss) serves as the intrinsic exploration reward.
* **Inverse Dynamics Model:** Predicts action $\hat{a}_t$ from $(\phi(s_t), \phi(s_{t+1}))$ to ensure feature invariance against uncontrollable environment noise.
* **Running Normalizer:** Online running mean and variance normalizer for stabilizing intrinsic reward scales.

### 3. Dueling DQN (`dqn.py`) & Prioritized Replay (`prioritized_replay.py`)
* **Dueling Architecture:** Decomposes the Q-value into state-value $V(s)$ and action advantage streams $A(s, a)$:
  $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)$$
* **Prioritized Experience Replay (PER):** Proportional Sum-Tree implementation with importance-sampling weights (annealing $\beta$ and exponent $\alpha$).

---

## 📂 Repository Structure

```text
.
├── actor_critic.py        # A2C network with 4-Conv layers, GRUCell, and GAE loss
├── curiosity.py           # Forward dynamics Curiosity Module with running reward normalization
├── curiosity_inverse.py   # Curiosity Module with joint Forward and Inverse dynamics models
├── dqn.py                 # Dueling DQN architecture (Value & Advantage streams)
├── prioritized_replay.py  # Prioritized Experience Replay Buffer using SumTree
├── train.py               # Training pipeline for Actor-Critic + ICM on Montezuma's Revenge
├── train_dqn.py           # Training pipeline for Dueling DQN with PER
├── evaluate.py            # Evaluation and visualization script for trained Actor-Critic agents
├── evaluate_dqn.py        # Evaluation script for trained Dueling DQN models
├── plot_graphics.ipynb    # Jupyter notebook for plotting rewards, losses, and steps
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/giadap00/Montezuma-Revenge-RL.git
cd Montezuma-Revenge-RL
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Training Actor-Critic with Curiosity (ICM)
```bash
python train.py
```
*Logs metrics (`rewards.npy`, `intrinsic_rewards.npy`, `icm_loss.npy`, `a2c_loss.npy`) and checkpoints every 50 episodes to `./results`.*

### Training Dueling DQN with PER
```bash
python train_dqn.py
```

### Evaluating Trained Agents
Evaluate the Actor-Critic agent with real-time rendering:
```bash
python evaluate.py --model_path results/model.pth --episodes 10 --render
```

Evaluate the Dueling DQN agent:
```bash
python evaluate_dqn.py --model_path results/dqn_model.pth --episodes 10 --render
```

### Plotting Training Curves
Open `plot_graphics.ipynb` in Jupyter Notebook or Google Colab to visualize training rewards, intrinsic exploration curves, and loss trends.

---

## 📄 License & Academic Integrity
Developed as part of the Reinforcement Learning course projects at Sapienza University of Rome (2025/2026).
