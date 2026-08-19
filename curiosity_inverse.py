import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class FeatureEncoder(nn.Module):
    """Enhanced feature encoder with larger network"""

    def __init__(self, input_channels=4, feature_dim=512):
        super(FeatureEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        features = F.relu(self.fc1(x))
        return features

class InverseDynamics(nn.Module):
    """Inverse dynamics model to predict action from (state, next_state)."""
    
    def __init__(self, feature_dim=512, action_dim=18, hidden_dim=512):
        super(InverseDynamics, self).__init__()
        self.fc1 = nn.Linear(2 * feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)   

    def forward(self, state_features, next_state_features):
        x = torch.cat([state_features, next_state_features], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits   

class ForwardDynamics(nn.Module):
    """Forward dynamics model with BatchNorm for stability"""

    def __init__(self, feature_dim=512, action_dim=18, hidden_dim=512):
        super(ForwardDynamics, self).__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(0.1)  

    def forward(self, state_features, action):
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=self.action_dim).float()

        x = torch.cat([state_features, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        next_features = self.fc3(x)
        return next_features

class CuriosityModule:
    """Curiosity module enhanced with inverse dynamics."""

    def __init__(self, state_shape, action_dim, device="cuda", learning_rate=0.001):
        self.device = device
        self.feature_dim = 512
        self.action_dim = action_dim

        # Initialize networks
        self.encoder = FeatureEncoder(input_channels=state_shape[0], feature_dim=self.feature_dim).to(device)
        self.forward_dynamics = ForwardDynamics(feature_dim=self.feature_dim, action_dim=action_dim).to(device)
        self.inverse_dynamics = InverseDynamics(feature_dim=self.feature_dim, action_dim=action_dim).to(device)

        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.forward_opt = torch.optim.Adam(self.forward_dynamics.parameters(), lr=learning_rate)
        self.inverse_opt = torch.optim.Adam(self.inverse_dynamics.parameters(), lr=learning_rate)

        # Reward normalization
        self.reward_normalizer = RunningNormalizer(scale=2.0)
        self.reward_memory = deque(maxlen=1000)

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity reward based on forward dynamics."""
        with torch.no_grad():
            state_features = self.encoder(state)
            next_state_features = self.encoder(next_state)
            predicted_features = self.forward_dynamics(state_features, action)

            # Forward dynamics loss (L1 + L2 error)
            l1_error = F.l1_loss(predicted_features, next_state_features, reduction='none').mean(dim=1)
            l2_error = F.mse_loss(predicted_features, next_state_features, reduction='none').mean(dim=1)
            forward_error = (l1_error + l2_error) / 2

            # Normalize reward
            reward = self.reward_normalizer(forward_error.cpu().numpy())
            self.reward_memory.extend(reward)

        return torch.FloatTensor(reward).to(self.device)

    def update(self, state, action, next_state):
        """Update forward and inverse dynamics models."""
        state_features = self.encoder(state)
        next_state_features = self.encoder(next_state)

        # Forward dynamics loss
        predicted_features = self.forward_dynamics(state_features, action)
        l1_loss = F.l1_loss(predicted_features, next_state_features.detach())
        l2_loss = F.mse_loss(predicted_features, next_state_features.detach())
        forward_loss = l1_loss + l2_loss

        # Inverse dynamics loss (cross-entropy on predicted action)
        predicted_actions = self.inverse_dynamics(state_features, next_state_features)
        inverse_loss = F.cross_entropy(predicted_actions, action.long())

        # Total loss (weighted sum)
        total_loss = forward_loss + 0.1 * inverse_loss

        # Optimize models
        self.encoder_opt.zero_grad()
        self.forward_opt.zero_grad()
        self.inverse_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), max_norm=1.0)
        self.encoder_opt.step()
        self.forward_opt.step()
        self.inverse_opt.step()

        return total_loss.item()

class RunningNormalizer:
    """Enhanced reward normalizer"""

    def __init__(self, epsilon=1e-8, scale=2.0):
        self.mean = 0
        self.std = 1
        self.count = 0
        self.epsilon = epsilon
        self.scale = scale

    def __call__(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = np.mean(x)
            self.std = np.std(x)
        else:
            old_mean = self.mean
            self.mean = old_mean + (np.mean(x) - old_mean) / self.count
            self.std = np.sqrt(self.std ** 2 + ((x - old_mean) * (x - self.mean)).mean())

        normalized = (x - self.mean) / (self.std + self.epsilon)
        return normalized * self.scale 