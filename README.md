# Curiosity-Driven Exploration by Self-Supervised Prediction in Montezuma's Revenge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Atari-0298c3.svg)](https://gymnasium.farama.org/)
[![ALE](https://img.shields.io/badge/ALE-MontezumaRevenge--v5-green.svg)](https://ale.farama.org/)

This project implements and compares Deep Reinforcement Learning (DRL) algorithms with **Curiosity-Driven Intrinsic Exploration** to solve **Montezuma's Revenge** (`ALE/MontezumaRevenge-v5`), one of the hardest sparse-reward environments in the Atari suite.

---

## 🎮 The Challenge & Environment
In *Montezuma's Revenge*, the agent must navigate a complex multi-room fortress, dodge lethal enemies (skulls, lasers), collect keys, and reach treasures under extremely **sparse extrinsic rewards**. Standard RL algorithms relying solely on external feedback fail to discover rewarding trajectories. 

### Preprocessing Pipeline:
1. **Resizing:** Scaled to $84 \times 84$ resolution.
2. **Grayscale Conversion:** Reduces computational overhead while preserving spatial geometry.
3. **Frame Skipping:** Frameskip of 4 with action repetition.
4. **Frame Stacking:** 4 consecutive frames stacked to capture motion and directional velocity.

---

## 🧠 Evaluated Architectures & Comparative Methods

### 1. Dueling DQN + Prioritized Experience Replay (PER) + Basic ICM
* **Dueling Streams:** Value function $V(s)$ and Action Advantage $A(s, a)$ combined via:
  $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)$$
* **Prioritized Replay (PER):** Sum-Tree based transition sampling with Importance Sampling weights $w_i = (N \cdot P(i))^{-\beta}$.
* **Exploration Module:** Forward dynamics prediction error with online Running Normalizer.
* **Findings:** Proved to be the **least effective** combination due to Q-value overestimation and instability in complex sparse environments, frequently causing the agent to get trapped in sub-optimal exploration loops.

### 2. A2C (Advantage Actor-Critic) + Forward Dynamics ICM
* **Actor-Critic Backbone:** Recurrent `GRUCell` core with Generalized Advantage Estimation (GAE):
  $$A(s_t, a_t) = \sum_{k=0}^{\infty} (\gamma \tau)^k \delta_{t+k}$$
* **Loss:** $L = L_{act} + L_{crit} - 0.01 \cdot L_{ent}$
* **Findings:** Policy gradients provided smoother and more stable policy updates compared to DQN. However, lacking an inverse dynamics model, it struggled to filter out non-actionable environment dynamics.

### 3. A2C + ICM with Joint Forward & Inverse Dynamics (Best Method 🏆)
* **Inverse Dynamics Model:** Predicts action $\hat{a}_t$ from $(\phi(s_t), \phi(s_{t+1}))$, forcing the latent space $\phi(s)$ to only encode features controllable by the agent.
* **Combined Loss:** Weighted sum of Forward Prediction Loss ($L_1 + L_2$) and Inverse Action Cross-Entropy Loss.
* **Findings:** **Most effective configuration**. Fast policy stabilization, superior exploration-exploitation trade-off, and consistent state discovery across training epochs.

---

## ⚙️ Hyperparameters

| Hyperparameter | Dueling DQN + ICM | A2C + ICM |
| :--- | :---: | :---: |
| **Discount Factor ($\gamma$)** | 0.99 | 0.99 |
| **GAE Parameter ($	au$)** | — | 1.0 |
| **Batch Size** | 32 | — |
| **Curiosity Learning Rate** | 0.001 | 0.001 |
| **$\epsilon$-Greedy ($\epsilon_{start} \to \epsilon_{end}$)** | $1.0 \to 0.1$ | — |
| **$\epsilon$-Decay Steps** | 200,000 | — |
| **Target Network Update Freq.** | 5,000 | — |
| **Frame Skip / Stack** | 4 / 4 | 4 / 4 |
| **Screen Resolution** | $84 \times 84$ | $84 \times 84$ |

---

## 📂 Repository Structure

```text
.
├── actor_critic.py         # Recurrent A2C network with 4-Conv layers, GRUCell & GAE
├── curiosity.py            # Basic Forward Dynamics Curiosity Module with running normalization
├── curiosity_inverse.py    # Enhanced Curiosity Module with Forward & Inverse Dynamics
├── dqn.py                  # Dueling DQN architecture (Value & Advantage streams)
├── prioritized_replay.py   # Prioritized Experience Replay (SumTree buffer)
├── train.py                # Training script for A2C + ICM
├── train_dqn.py            # Training script for Dueling DQN + PER + ICM
├── evaluate.py             # Evaluation and test runner for trained A2C models
├── evaluate_dqn.py         # Evaluation runner for trained Dueling DQN models
├── plot_graphics.ipynb     # Jupyter notebook for metrics visualization
├── RL_FinalProject.pdf     # Presentation slides with theoretical details and results
├── requirements.txt        # Python package dependencies
├── results/                # Evaluation metric arrays (.npy)
│   ├── rewards.npy
│   ├── intrinsic_rewards.npy
│   ├── extrinsic_rewards.npy
│   ├── a2c_loss.npy
│   └── icm_loss.npy
└── README.md               # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/giadap00/Montezuma-Revenge-RL.git
cd Montezuma-Revenge-RL
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training
```bash
# Train A2C with Inverse Dynamics ICM (Recommended)
python train.py

# Train Dueling DQN with PER
python train_dqn.py
```

### 3. Evaluation
```bash
python evaluate.py --model_path results/final_model.pth --episodes 10 --render
```

---

## 📚 References
* Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. *Curiosity-driven Exploration by Self-supervised Prediction*. ICML 2017.
