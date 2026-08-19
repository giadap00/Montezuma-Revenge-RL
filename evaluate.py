import gymnasium as gym
from gym.wrappers.frame_stack import FrameStack
from gymnasium.wrappers import AtariPreprocessing
from actor_critic import ActorCritic
import torch
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

    # Load Curiosity Module model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    agent = ActorCritic(input_dims, n_actions).to(device)
    
    
    # Load the model's state_dict
    checkpoint = torch.load(model_path)
    agent.load_state_dict(checkpoint)  # Load encoder weights
    agent.eval()

    total_rewards = []
    total_steps = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        hx = torch.zeros(1, 256).to(device)

        while not (done or truncated):
            with torch.no_grad():
                # Process the current state with the actor-critic model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _, _, hx = agent(state_tensor, hx)
                action = action.item()

            # Take the chosen action in the environment
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(np.asarray(next_state, dtype=np.float32) / 255.0).to(device)
            episode_reward += reward
            steps += 1

            

            state = next_state

        print(f"Episode {episode + 1} - Reward: {episode_reward}, Steps: {steps}")
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