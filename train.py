import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
import ale_py
from gym.wrappers.frame_stack import FrameStack
from gymnasium.wrappers import AtariPreprocessing
import gc

from actor_critic import ActorCritic 
from curiosity import CuriosityModule

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurations
ENV_NAME = 'ALE/MontezumaRevenge-v5'
NUM_EPISODES = 50000
GAMMA = 0.99
TAU = 1.0
RESULTS_DIR = "./results"
curiosity_learning_rate = 0.001
reward_scale = 2.0

# Create environment
env = gym.make(ENV_NAME, render_mode='rgb_array', frameskip=1)
env = AtariPreprocessing(env, noop_max=10, frame_skip=4, terminal_on_life_loss=False, screen_size=84, grayscale_obs=True, grayscale_newaxis=False)
env = FrameStack(env, num_stack=4)

num_actions = env.action_space.n
input_dims = env.observation_space.shape

# Neural Network
policy_net = ActorCritic(input_dims, num_actions, gamma=GAMMA, tau=TAU).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

# Curiosity Module
curiosity_module = CuriosityModule(state_shape=input_dims, action_dim=num_actions, device=device, learning_rate=curiosity_learning_rate)

# Tracking data
rewards_list = []
extrinsic_rewards = []
intrinsic_rewards = []
frames_per_reward = []
a2c_loss_list = []
icm_loss_list = []
step_list = []
cumulative_reward = 0

# Training loop
for episode in range(NUM_EPISODES):
    steps = 0
    state, _ = env.reset()
    gc.collect()
    state = torch.FloatTensor(np.asarray(state, dtype=np.float32) / 255.0).to(device)
    episode_reward = 0
    episode_intrinsic_reward = 0
    episode_extrinsic_reward = 0
    episode_frames = 0
    done = False
    hx = torch.zeros(1, 256).to(device)
    rewards, values, log_probs = [], [], []
    
    while not done:
        state_tensor = state.unsqueeze(0)
        action, value, log_prob, hx = policy_net(state_tensor, hx)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(np.asarray(next_state, dtype=np.float32) / 255.0).to(device)
        intrinsic_reward = curiosity_module.compute_intrinsic_reward(state_tensor, torch.tensor([action]).to(device), next_state.unsqueeze(0))
        icm_loss = curiosity_module.update(state_tensor, torch.tensor([action]).to(device), next_state.unsqueeze(0))
        total_reward = intrinsic_reward.cpu().numpy()[0] + reward
        rewards.append(total_reward)
        values.append(value)
        log_probs.append(log_prob)
        
        episode_reward += total_reward
        episode_extrinsic_reward += reward
        episode_intrinsic_reward += intrinsic_reward
        cumulative_reward += episode_extrinsic_reward
        
        if done or truncated:
            a2c_loss = policy_net.calc_cost(next_state, hx, done, rewards, values, log_probs)
            optimizer.zero_grad()
            a2c_loss.backward()
            optimizer.step()
            a2c_loss_list.append(a2c_loss)
        
        state = next_state
        steps += 1
        episode_frames += 1
        
    rewards_list.append(episode_reward)
    extrinsic_rewards.append(episode_extrinsic_reward)
    intrinsic_rewards.append(episode_intrinsic_reward)
    frames_per_reward.append(episode_frames)
    step_list.append(steps)
    icm_loss_list.append(icm_loss)
    
    print(f"Episode {episode}, Total Reward: {episode_reward}, Extrinsic Reward: {episode_extrinsic_reward}, Intrinsic Reward: {episode_intrinsic_reward.cpu().numpy()[0]} Steps: {steps}, A2C Loss: {a2c_loss}, ICM loss: {icm_loss}")
    
    if episode % 50 == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        model_path = os.path.join(RESULTS_DIR, f"a2c_model{episode}.pth")
        torch.save(policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        np.save(os.path.join(RESULTS_DIR, "rewards.npy"), rewards_list)
        np.save(os.path.join(RESULTS_DIR, "extrinsic_rewards.npy"), extrinsic_rewards)
        np.save(os.path.join(RESULTS_DIR, "intrinsic_rewards.npy"), intrinsic_rewards)
        np.save(os.path.join(RESULTS_DIR, "steps.npy"), step_list)
        np.save(os.path.join(RESULTS_DIR, "frames_per_reward.npy"), frames_per_reward)
        np.save(os.path.join(RESULTS_DIR, "a2c_loss.npy"), [a2c_loss.detach().numpy() for a2c_loss in a2c_loss_list])
        np.save(os.path.join(RESULTS_DIR, "icm_loss.npy"), icm_loss_list)
        gc.collect()
        
# Save final results
os.makedirs(RESULTS_DIR, exist_ok=True)
model_path = os.path.join(RESULTS_DIR, "model.pth")
torch.save(policy_net.state_dict(), model_path)
print(f"Model saved to {model_path}")

np.save(os.path.join(RESULTS_DIR, "rewards.npy"), rewards_list)
np.save(os.path.join(RESULTS_DIR, "extrinsic_rewards.npy"), extrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "intrinsic_rewards.npy"), intrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "steps.npy"), step_list)
np.save(os.path.join(RESULTS_DIR, "frames_per_reward.npy"), frames_per_reward)
np.save(os.path.join(RESULTS_DIR, "a2c_loss.npy"), [a2c_loss.detach().numpy() for a2c_loss in a2c_loss_list])
        
