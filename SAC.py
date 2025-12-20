import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Normal
import numpy as np 
import gymnasium as gym 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0 
        self.size = 0
       
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.state[self.ptr]      = state 
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward 
        self.next_state[self.ptr] = next_state
        self.done[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )
    

    def __len__(self):
        return self.size
    

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.apply(initialize_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, hidden_dim=256):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.apply(initialize_weights)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        X_T = normal.rsample()
        Y_t = torch.tanh(X_T)
        action = Y_t

        log_prob = normal.log_prob(X_T)
        log_prob -= torch.log(1 - Y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
    
class SACAgent:
    def __init__(self, state_dim, action_dim, gamma = 0.99, tau = 0.005, alpha = 0.2, lr= 3e-4):
        self.gamma = gamma 
        self.tau  = tau
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q2 = Critic(state_dim, action_dim).to(device)

        self.q1_target = Critic(state_dim, action_dim).to(device)
        self.q2_target = Critic(state_dim, action_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.critic_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr = lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
           mean, _ = self.actor(state)
           action = torch.tanh(mean).detach()
           return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
        
    def update(self, replay_buffer, batch_size):
        state_b, action_b, reward_b, next_state_b, match_b = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state_b)

            q1_next_target = self.q1_target(next_state_b, next_action)
            q2_next_target = self.q2_target(next_state_b, next_action)

            min_qf_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
            next_q_value = reward_b + (1 - match_b) * self.gamma * (min_qf_next_target)

        q1 = self.q1(state_b, action_b)
        q2 = self.q2(state_b, action_b)

        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(state_b)

        q1_pi = self.q1(state_b, pi)
        q2_pi = self.q2(state_b, pi)

        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()
        self.alpha = self.alpha.item()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


def train():
    env_name = "Humanoid-v5"
    env = gym.make(env_name)

    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.ClipAction(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)

    memory = ReplayBuffer(1_000_000, state_dim, action_dim)
    max_steps = 3000000
    batch_size = 256
    start_steps = 10000

    total_steps = 0 
    episode = 0 

    while total_steps < max_steps:
        state, _ = env.reset(seed=42*episode) 
        episode_reward = 0 
        done = False

        while not done:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else: 
                action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            mask = 1 if terminated else 0 

            memory.push(state, action, reward, next_state, mask)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= start_steps:
                agent.update(memory, batch_size)
        episode += 1

        if (episode % 100 == 0):
            print(f"Episode: {episode}, Total Steps: {total_steps}, Episode Reward: {episode_reward}")
            
            if (episode % 1000 == 0):
                torch.save(agent.actor.state_dict(), f"sac{episode}.pth")
    torch.save(agent.actor.state_dict(), f"sac_final.pth")
    
    env.close() 

def play(model_path):
    env = gym.make("Humanoid-v5", render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim,action_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
     
    num_test_episodes = 10
    
    for episode in range(num_test_episodes):
        state, _ = env.reset(seed=42*episode)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mean, _ = actor(state_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
            
        print(f"Episode {episode+1}: Reward {total_reward:.2f}, Steps {steps}")
        
    env.close()

if __name__ == "__main__":
    play("sac_final.pth")

