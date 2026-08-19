import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
from pathlib import Path
import ale_py
from gym.wrappers.frame_stack import FrameStack
from gymnasium.wrappers import AtariPreprocessing
from dqn import DuelingDQN
import gc

from prioritized_replay import PrioritizedReplayBuffer
from curiosity import CuriosityModule

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurations
ENV_NAME = 'ALE/MontezumaRevenge-v5'
NUM_EPISODES = 50000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 200000
TARGET_UPDATE_FREQUENCY = 5000
RESULTS_DIR = "./results"
# Curiosity Module Configurations
curiosity_learning_rate = 0.001
reward_scale = 2.0

# Create environment
env = gym.make(ENV_NAME, render_mode='rgb_array', frameskip=1)
env = AtariPreprocessing(env, noop_max=10, frame_skip=4, terminal_on_life_loss=False, screen_size=84, grayscale_obs=True, grayscale_newaxis=False)
env = FrameStack(env, num_stack=4)

num_actions = env.action_space.n
input_channels = env.observation_space.shape[0]

# Neural Networks
policy_net =  DuelingDQN(input_channels= input_channels , num_actions= num_actions).to(device)
target_net =  DuelingDQN(input_channels= input_channels , num_actions= num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and replay buffer
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
replay_buffer = PrioritizedReplayBuffer(capacity=100000)
obv_shape = env.observation_space.shape

curiosity_module = CuriosityModule(state_shape=obv_shape, action_dim=env.action_space.n, device=device, learning_rate=curiosity_learning_rate)

# Epsilon-greedy schedule
def epsilon_by_step(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY_STEPS)

# Tracking data for analysis
rewards_list = []
extrinsic_rewards = []
intrinsic_rewards = []
frames_per_reward = []
dqn_loss_list = []
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
    truncated = False
    while not (done or truncated):
        epsilon = epsilon_by_step(steps)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        next_state, reward, done, truncated, info= env.step(action)
        next_state = torch.FloatTensor(np.asarray(next_state, dtype=np.float32) / 255.0).to(device)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        intrinsic_reward = curiosity_module.compute_intrinsic_reward(state_tensor, torch.tensor([action]).to(device), next_state_tensor)
        icm_loss = curiosity_module.update(state_tensor, torch.tensor([action]).to(device), next_state.unsqueeze(0))
        total_reward = intrinsic_reward.cpu().numpy()[0]
        replay_buffer.push(state.cpu().numpy(), action, total_reward, next_state.cpu().numpy(), done or truncated)
        

        episode_reward += total_reward
        episode_extrinsic_reward += reward
        episode_intrinsic_reward += intrinsic_reward
        cumulative_reward += episode_extrinsic_reward


        if len(replay_buffer) > BATCH_SIZE:
            batch, indices, weights = replay_buffer.sample(BATCH_SIZE)
             
            states = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
            actions = torch.LongTensor([t.action for t in batch]).to(device)
            rewards = torch.FloatTensor([t.reward for t in batch]).to(device)
            next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
            dones = torch.FloatTensor([t.done for t in batch]).to(device)
            
            # Compute Q values with double Q-learning
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_actions = policy_net(next_states).max(1)[1]
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_q = rewards + GAMMA * next_q * (1 - dones)

            # Compute loss with prioritized replay
            dqn_loss = (torch.tensor(weights, device=device) *
                            (current_q.squeeze() - expected_q.detach()) ** 2).mean()

            optimizer.zero_grad()
            dqn_loss.backward()
            optimizer.step()
            dqn_loss_list.append(dqn_loss)

        state = next_state
        steps += 1
        episode_frames += 1
        
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

    
    rewards_list.append(episode_reward)
    extrinsic_rewards.append(episode_extrinsic_reward)
    intrinsic_rewards.append(episode_intrinsic_reward)
    frames_per_reward.append(episode_frames)  
    step_list.append(steps)
    icm_loss_list.append(icm_loss)

    if episode % 50 == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        model_path = os.path.join(RESULTS_DIR, f"dqn_model{episode}.pth")
        torch.save(policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        np.save(os.path.join(RESULTS_DIR, "rewards.npy"), rewards_list)
        np.save(os.path.join(RESULTS_DIR, "extrinsic_rewards.npy"), extrinsic_rewards)
        np.save(os.path.join(RESULTS_DIR, "intrinsic_rewards.npy"), intrinsic_rewards)
        np.save(os.path.join(RESULTS_DIR, "steps.npy"), step_list)
        np.save(os.path.join(RESULTS_DIR, "frames_per_reward.npy"), frames_per_reward)
        np.save(os.path.join(RESULTS_DIR, "dqn_loss.npy"), [dqn_loss.detach().numpy() for dqn_loss in dqn_loss_list])
        np.save(os.path.join(RESULTS_DIR, "icm_loss.npy"), icm_loss_list)
    print(f"Episode {episode}, Total Reward: {episode_reward}, Extrinsic Reward: {episode_extrinsic_reward}, Intrinsic Reward: {episode_intrinsic_reward.cpu().numpy()[0]} Steps: {steps}, DQN Loss: {dqn_loss}, ICM loss: {icm_loss}")
    
# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)

model_path = os.path.join(RESULTS_DIR, "dqn_model.pth")
torch.save(policy_net.state_dict(), model_path)
print(f"Model saved to {model_path}")

np.save(os.path.join(RESULTS_DIR, "rewards.npy"), rewards_list)
np.save(os.path.join(RESULTS_DIR, "extrinsic_rewards.npy"), extrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "intrinsic_rewards.npy"), intrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "steps.npy"), step_list)
np.save(os.path.join(RESULTS_DIR, "frames_per_reward.npy"), frames_per_reward)
np.save(os.path.join(RESULTS_DIR, "dqn_loss.npy"), [dqn_loss.detach().numpy() for dqn_loss in dqn_loss_list])
np.save(os.path.join(RESULTS_DIR, "icm_loss.npy"), icm_loss_list)
gc.collect()
