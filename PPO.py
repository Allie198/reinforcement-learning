import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
env_name = "LunarLander-v3"
learning_rate = 3e-4
gamma = 0.99 
lmbda = 0.95
eps_clip = 0.2 
K_Epochs = 10 
T_horizon = 2048
timesteps = 300000

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), self.critic(state)

    def evaluate_action(self, state, action):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy
    
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()  

    def update(self, memory):
        rewards = [] 
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(device)

        for _ in range(K_Epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate_action(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach() 

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = [] 
        self.logprobs = []
        self.rewards = [] 
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train():
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim)

    print(f"Training on {env_name}")

    time_step = 0
    running_reward = 0
    
    state, _ = env.reset() 

    for i in range(1, timesteps + 1):
        action, log_prob, val = ppo.policy_old.act(state)
        next_state, reward, done , truncated, _ = env.step(action)

        memory.states.append(torch.from_numpy(state).float())
        memory.actions.append(torch.tensor(action))
        memory.logprobs.append(log_prob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done or truncated)

        state = next_state
        running_reward += reward
        time_step += 1


        if time_step % T_horizon == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0

        if done or truncated:
            state, _ = env.reset()

        if i % 10000 == 0:
            print(f"Step: {i} \t Avg Reward: {running_reward/10000:.2f}")
            running_reward = 0
            torch.save(ppo.policy.state_dict(), 'ppo_lunar_model.pth')

def play():
    env = gym.make(env_name, render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim).to(device)
    filename = 'ppo_lunar_model.pth'
    
    try:
        policy.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        print(f"Model loaded: {filename}")
    except FileNotFoundError:
        train()

    policy.eval()
    
    episodes = 20
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            with torch.no_grad():
                action, _, _ = policy.act(state)
            
            state, reward, done, truncated, _ = env.step(action)
            score += reward
            
            if done or truncated:
                break
                
        print(f"Episode {ep+1} | Score : {score:.2f}")


    env.close()

if __name__ == '__main__':
    play()
