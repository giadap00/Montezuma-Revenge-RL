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


class ForwardDynamics(nn.Module):
    """Enhanced forward dynamics model"""

    def __init__(self, feature_dim=512, action_dim=18, hidden_dim=512):
        super(ForwardDynamics, self).__init__()

        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(0.2)
        self.action_dim = action_dim

    def forward(self, state_features, action):
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=self.action_dim).float()

        x = torch.cat([state_features, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        next_features = self.fc3(x)
        return next_features


class CuriosityModule(nn.Module):  
    """Enhanced curiosity module with stronger rewards"""

    def __init__(self, state_shape, action_dim, device="cuda", learning_rate=0.001):
        super(CuriosityModule, self).__init__()  
        self.device = device
        self.feature_dim = 512
        self.action_dim = action_dim

        # Initialize networks
        self.encoder = FeatureEncoder(input_channels=state_shape[0],
                                      feature_dim=self.feature_dim).to(device)
        self.forward_dynamics = ForwardDynamics(feature_dim=self.feature_dim,
                                                action_dim=action_dim).to(device)

        # Optimizers with higher learning rate
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.forward_opt = torch.optim.Adam(self.forward_dynamics.parameters(), lr=learning_rate)

        # Enhanced reward normalization
        self.reward_normalizer = RunningNormalizer(scale=2.0)   
        self.reward_memory = deque(maxlen=1000)   

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute enhanced intrinsic reward"""
        with torch.no_grad():
            state_features = self.encoder(state)
            next_state_features = self.encoder(next_state)

            predicted_features = self.forward_dynamics(state_features, action)

            # Compute prediction error with L1 and L2 components
            l1_error = F.l1_loss(predicted_features, next_state_features, reduction='none').mean(dim=1)
            l2_error = F.mse_loss(predicted_features, next_state_features, reduction='none').mean(dim=1)
            prediction_error = (l1_error + l2_error) / 2

            # Apply enhanced normalization
            reward = self.reward_normalizer(prediction_error.cpu().numpy())
            self.reward_memory.extend(reward.tolist())

        return torch.FloatTensor(reward).to(self.device)

    def update(self, state, action, next_state):
        """Update dynamics model with increased learning"""
        state_features = self.encoder(state)
        next_state_features = self.encoder(next_state)

        predicted_features = self.forward_dynamics(state_features, action)

        # Enhanced loss computation
        l1_loss = F.l1_loss(predicted_features, next_state_features.detach())
        l2_loss = F.mse_loss(predicted_features, next_state_features.detach())
        forward_loss = l1_loss + l2_loss

        self.encoder_opt.zero_grad()
        self.forward_opt.zero_grad()
        forward_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), max_norm=1.0)

        self.encoder_opt.step()
        self.forward_opt.step()

        return forward_loss.item()


class RunningNormalizer:
    """Enhanced reward normalizer with correct incremental updates."""

    def __init__(self, epsilon=1e-8, scale=2.0):
        self.mean = 0
        self.var = 1  
        self.count = 1e-4
        self.epsilon = epsilon
        self.scale = scale

    def update(self, x):
        """Aggiorna la media e la varianza in modo incrementale"""
        x = np.asarray(x) 
        batch_count = len(x)
        batch_mean = np.mean(x)
        batch_var = np.var(x)
 
        new_count = self.count + batch_count
 
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / new_count)
        self.var = ((self.var * self.count) + (batch_var * batch_count) + 
                    (delta**2 * (self.count * batch_count / new_count))) / new_count
        self.count = new_count 

    def __call__(self, x):
        """Normalizza l'input con i valori aggiornati di mean e std"""
        self.update(x)   
        normalized = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return normalized * self.scale   
