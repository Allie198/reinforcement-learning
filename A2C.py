import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import gymnasium as gym 

class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.shared_layers = nn.Sequential(

            nn.Linear(input_shape,128),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)

        action_logits = self.actor_head(x)
        value = self.critic_head(x)

        probs = F.softmax(action_logits, dim=-1)


        return probs, value  


def select_action(network, state):
    state = torch.from_numpy(state).float().unsqueeze(0)    
    probs, state_value = network(state)

    M = torch.distributions.Categorical(probs)

    action = M.sample() 
    log_prob = M.log_prob(action)
    entropy = M.entropy()

    return action.item(), log_prob, entropy, state_value


def complete_returns(next_value, rewards, masks, gamma = 0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0,R)
    return returns 


def train(max_episodes=1000, n_steps=10, entropy_coefficient=0.001):
 
    env = gym.make("CartPole-v1")
    model = A2C(input_shape=env.observation_space.shape[0], n_actions=env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

 
    
    for episode in range(max_episodes):
        state, _ = env.reset() 
        done = False 
        score = 0 

        while not done: 
            log_probs = [] 
            entropies = []
            values    = []
            rewards   = [] 
            masks     = []


            for _ in range(n_steps):
                action, log_prob, entropy, value = select_action(model, state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)

                rewards.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([0.0 if done else 1.0], dtype=torch.float))

                state = next_state
                score += reward

                if done:
                    break 
            
            next_value = 0
            
            if not done:
                _, next_value = model(torch.from_numpy(next_state).float().unsqueeze(0))

            returns = complete_returns(next_value, rewards, masks)

            log_probs = torch.stack(log_probs).squeeze(-1)
            entropies = torch.stack(entropies).squeeze(-1)
            values = torch.cat(values).squeeze(-1)
            returns = torch.cat(returns).detach().squeeze(-1)
            advantages = returns - values 

            critic_loss = advantages.pow(2).mean() 
            actor_loss = -(log_probs * advantages.detach()).mean()

            total_loss = critic_loss + actor_loss - entropy_coefficient * entropies.mean()

            optimizer.zero_grad()
            total_loss.backward() 
            optimizer.step()
        print(f"Episode: {episode}, Score: {score}")
 
    env.close()
    return model 
    
def play(model, episodes=100, deterministic = True):
    env = gym.make("CartPole-v1", render_mode = "human")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False 
        score = 0 

        while not done: 
            state_torch = torch.from_numpy(state).float().unsqueeze(0)

            with torch.no_grad():
                probs, _ = model(state_torch)

            if deterministic:
               action = torch.argmax(probs, dim=-1).item() 

            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        print(f"Episode {ep+1}, Score {score}")


if __name__ == "__main__":
    model = train()
    play(model)



