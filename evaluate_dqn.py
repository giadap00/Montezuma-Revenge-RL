import gymnasium as gym
from gym.wrappers.frame_stack import FrameStack
from gymnasium.wrappers import AtariPreprocessing
import torch
from dqn import DuelingDQN
import numpy as np
import argparse
import ale_py

def evaluate(model_path, episodes=10, render=True):
    # Create environment
    env = gym.make('ALE/MontezumaRevenge-v5',
                   frameskip=4,
                   render_mode='human' if render else None)
    
    env = AtariPreprocessing(env, 
                           grayscale_obs=True,
                           frame_skip=1,
                           scale_obs=True)
    env = FrameStack(env, num_stack=4)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = DuelingDQN(input_channels=input_channels, num_actions=n_actions).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    total_rewards = []
    total_steps = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        positions = set()

        while not (done or truncated):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).argmax().item()

            state, reward, done, truncated, info = env.step(action)
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32) / 255.0).to(device)
            episode_reward += reward
            steps += 1

        print(f"Episode {episode + 1} - Reward: {episode_reward}, Steps: {steps}, Unique Positions: {len(positions)}")
        total_rewards.append(episode_reward)
        total_steps.append(steps)

    env.close()

    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.2f}")
    print(f"Max Reward: {max(total_rewards)}")
    print(f"Max Steps: {max(total_steps)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained Montezuma\'s Revenge agent')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    
    args = parser.parse_args()
    evaluate(args.model_path, args.episodes, args.render)
