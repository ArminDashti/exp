import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import deque
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.Q1(x), self.Q2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.latent_buffer = deque(maxlen=10000)
    def push(self, state, action, reward, next_state, done, intrinsic):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward + intrinsic, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

class TD3Agent:
    def __init__(self):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.autoencoder = Autoencoder(state_dim).to(device)
        self.cluster_model = MiniBatchKMeans(n_clusters=10, random_state=SEED)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(int(1e6))
        self.ae_loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.exploration_noise = 0.1
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self.alpha = 1.0
        self.alpha_decay = 0.9995
        self.cluster_update_freq = 200
        self.policy_update_freq = 2
    def update_targets(self):
        for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
    def get_intrinsic_reward(self, latent):
        latent = latent.reshape(1, -1)
        # Only update clusters when there's enough data in the latent buffer
        if len(self.buffer.latent_buffer) < self.cluster_model.n_clusters:
            return self.alpha
        latent_buffer_np = np.array(self.buffer.latent_buffer)
        # Update the cluster model with the entire latent buffer
        self.cluster_model.partial_fit(latent_buffer_np)
        cluster_label = self.cluster_model.predict(latent)[0]
        counts = np.sum(self.cluster_model.predict(latent_buffer_np) == cluster_label)
        return self.alpha / (np.sqrt(counts) + 1e-6)
    def train(self, episodes=2000):
        self._pretrain_autoencoder(1000)
        for episode in range(episodes):
            state, _ = env.reset(seed=SEED)
            episode_extrinsic = 0
            episode_intrinsic = 0
            self.alpha *= self.alpha_decay
            for t in range(1000):
                noise_scale = self.exploration_noise * (1 - episode / episodes)
                action = self._select_action(state, noise_scale)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ae_loss = self._update_autoencoder(state)
                with torch.no_grad():
                    _, latent = self.autoencoder(torch.FloatTensor(state).to(device))
                latent_np = latent.cpu().numpy().flatten()
                self.buffer.latent_buffer.append(latent_np)
                intrinsic = self.get_intrinsic_reward(latent_np)
                self.buffer.push(state, action, reward, next_state, done, intrinsic)
                episode_extrinsic += reward
                episode_intrinsic += intrinsic
                state = next_state
                if len(self.buffer.buffer) >= self.batch_size:
                    self._update_critic()
                    if t % self.policy_update_freq == 0:
                        self._update_actor()
                        self.update_targets()
                if done:
                    break
            print(f"Episode {episode}: Extrinsic: {episode_extrinsic:.1f} | Intrinsic: {episode_intrinsic:.1f} | AE Loss: {ae_loss:.4f}")
    def _pretrain_autoencoder(self, steps):
        print("Pretraining autoencoder...")
        while len(self.buffer.buffer) < 5000:
            state, _ = env.reset(seed=SEED)
            for _ in range(1000):
                action = env.action_space.sample()
                next_state, reward, done, _, _ = env.step(action)
                self.buffer.push(state, action, reward, next_state, done, 0)
                state = next_state if not done else env.reset()[0]
        for _ in range(steps):
            states = self.buffer.sample(self.batch_size)[0]
            states = torch.FloatTensor(states).to(device)
            recon, _ = self.autoencoder(states)
            loss = self.ae_loss_fn(recon, states)
            self.ae_optimizer.zero_grad()
            loss.backward()
            self.ae_optimizer.step()
    def _select_action(self, state, noise_scale):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy().flatten()
        noise = np.random.normal(0, noise_scale, size=action_dim)
        return np.clip(action + noise, -1, 1)
    def _update_critic(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_actions) * self.target_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_value = rewards + (1 - dones) * self.gamma * target_Q
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q1, target_value) + nn.MSELoss()(current_Q2, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
    def _update_actor(self):
        states = self.buffer.sample(self.batch_size)[0]
        states = torch.FloatTensor(states).to(device)
        q_value = self.critic(states, self.actor(states))[0]
        actor_loss = -q_value.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
    def _update_autoencoder(self, state):
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        recon, _ = self.autoencoder(state_tensor)
        loss = self.ae_loss_fn(recon, state_tensor)
        self.ae_optimizer.zero_grad()
        loss.backward()
        self.ae_optimizer.step()
        return loss.item()

if __name__ == "__main__":
    agent = TD3Agent()
    agent.train(episodes=2000)
    env.close()
