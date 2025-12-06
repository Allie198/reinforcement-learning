import random
import collections
import numpy as np
import cv2
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import os

gym.register_envs(ale_py)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[32:194, :]
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

class Dueling_DCQN(nn.Module):
    def __init__(self, obs_channels, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)

        )

    def forward(self, z):
        z = self.conv(z)
        z = z.view(z.size(0), -1)

        value = self.value(z)
        advantage = self.advantage(z)

        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals


def select_action(net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
        with torch.no_grad():
            q_vals = net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
        return int(act_v.item())

def train(env_id="ALE/Breakout-v5", num_frames=200000, batch_size=32, gamma=0.99, replay_size=100000,
          learning_rate=2e-4, sync_target_frames=1000, replay_start_size=50000, epsilon_start=1.0,
          epsilon_final=0.01, epsilon_decay_last_frame=100000, frame_stack=4, model_path="dcqn.pth"):

    env = gym.make(env_id, render_mode=None)
    n_actions = env.action_space.n
    net = Dueling_DCQN(frame_stack, n_actions).to(DEVICE)
    target = Dueling_DCQN(frame_stack, n_actions).to(DEVICE)
    target.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(replay_size)

    def calc_epsilon(frame_idx):
        return max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_last_frame)

    obs, _ = env.reset()
    frame = preprocess(obs)
    state = np.stack([frame] * frame_stack, axis=0)

    total_reward = 0.0
    episode_rewards = []
    frame_idx = 0

    while frame_idx < num_frames:
        frame_idx += 1
        epsilon = calc_epsilon(frame_idx)

        action = select_action(net, state, epsilon, n_actions)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_frame = preprocess(next_obs)
        next_state = np.roll(state, shift=-1, axis=0)
        next_state[-1] = next_frame

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            obs, info = env.reset()
            if info.get("lives") is not None:
                obs, _, _, _ ,_ = env.step(1)

            frame = preprocess(obs)
            state = np.stack([frame] * frame_stack, axis=0)
            episode_rewards.append(total_reward)
            print(f"Frame : {frame_idx} Episode reward : {total_reward} Epsilon : {epsilon:.3f}")
            total_reward = 0.0

        if len(buffer) < replay_start_size:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_v = torch.tensor(states, dtype=torch.float32, device=DEVICE) / 255.0
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=DEVICE) / 255.0
        actions_v = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        done_mask = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        q_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_online = net(next_states_v)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = target(next_states_v).gather (1, next_actions.unsqueeze(-1)).squeeze(1)
            next_q_target[done_mask] = 0.0
            expected_q_values = rewards_v + gamma * next_q_target

        loss = nn.SmoothL1Loss()(q_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 1.0)
        optimizer.step()

        if frame_idx % sync_target_frames == 0:
            target.load_state_dict(net.state_dict())
            print(f"Target network updated at frame {frame_idx}")


    torch.save(net.state_dict(), model_path)

    env.close()
    return net, episode_rewards


def start_with_fire(env, fire_action=1, steps=3):
    obs, info = env.reset()
    for _ in range(steps):
        obs, _, terminated, truncated, info = env.step(fire_action)
        if terminated or truncated:
            obs, info = env.reset()
    return obs,info

def play_trained_model(model_path="dcqn.pth", env_id="ALE/Breakout-v5", frame_stack=4):
    env = gym.make(env_id, render_mode="human")
    n_actions = env.action_space.n
    net = Dueling_DCQN(frame_stack, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    obs, info = start_with_fire(env)
    lives = info.get("lives",0)

    frame = preprocess(obs)
    state = np.stack([frame] * frame_stack, axis=0)

    total_reward = 0.0
    while True:
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
        with torch.no_grad():
            q_vals = net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
            action = int(act_v.item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        new_lives = info.get("lives",lives)
        life_lost = new_lives < lives
        lives = new_lives

        if life_lost and not done:
            for _ in range(3):
                next_obs, _, terminated2, truncated2, info = env.step(1)  # 1 = FIRE
                if terminated2 or truncated2:
                    done = True
                    break

        total_reward += reward

        next_frame = preprocess(next_obs)
        next_state = np.roll(state, shift=-1, axis=0)
        next_state[-1] = next_frame
        state = next_state

        if done:
            print("Episode reward:", total_reward)
            total_reward = 0.0
            obs, _ = start_with_fire(env)
            lives = info.get("lives",0)
            frame = preprocess(obs)
            state = np.stack([frame] * frame_stack, axis=0)

if __name__ == '__main__':
    if os.path.exists('/dcqn.pth'):
        play_trained_model()
    else:
        train()
        play_trained_model()

