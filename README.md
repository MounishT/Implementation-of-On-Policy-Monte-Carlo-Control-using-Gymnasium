# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description


FrozenLake-v1 is a reinforcement learning environment provided by Gymnasium. It contains a grid with a starting position, frozen surfaces, holes, and a goal. The agent moves across the grid and receives a reward when it successfully reaches the goal.

For this experiment, is_slippery=False is used so that the environment is deterministic and the agent can learn the optimal policy more easily.

The environment contains:

S – Starting state
F – Frozen tile
H – Hole
G – Goal

The standard 4 × 4 FrozenLake environment contains 16 states and 4 actions:

| Action | Meaning |
|--------|--------|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Start the FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning parameters α, γ, and ε.
4. Reset the environment and start a new episode.
5. Select an action using the epsilon-greedy policy.
6. Execute the action and observe the next state and reward.
7. Continue until the episode terminates.
8. Calculate the return G by processing the episode backwards.
9. Update the Q-value using the Monte Carlo update rule.
10. Reduce epsilon gradually after every episode.
11. Repeat the process for the specified number of episodes.
12. Calculate the state-value function from the learned Q-table.
13. Select the best action for every state to obtain the final policy.
14. Display the Q-table, state-value function, learned policy, average reward, and learning curve.

## Python Program


#### Monte Carlo Control


```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 20000
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))

returns_sum = {}
returns_count = {}

episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Policy
# -------------------------------------------------
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)   # Explore
    else:
        return np.argmax(Q[state])            # Exploit


# -------------------------------------------------
# Generate Episode
# -------------------------------------------------
def generate_episode():
    episode = []
    state, _ = env.reset()

    for _ in range(max_steps):

        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))
        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control (First-Visit)
# -------------------------------------------------
for episode_num in range(num_episodes):

    episode = generate_episode()

    G = 0
    visited = set()
    total_reward = 0

    # Backward return calculation
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = gamma * G + reward
        total_reward += reward

        if (state, action) not in visited:
            visited.add((state, action))

            if (state, action) not in returns_sum:
                returns_sum[(state, action)] = 0
                returns_count[(state, action)] = 0

            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1

            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

    episode_rewards.append(total_reward)

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract Policy & Value Function
# -------------------------------------------------
policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------
print("\nFinal Q-table:")
print(np.round(Q, 3))

print("\nState Value Function:")
print(np.round(state_values.reshape(4, 4), 3))


# Policy display
actions = {0: "L", 1: "D", 2: "R", 3: "U"}

policy_grid = np.array([actions[a] for a in policy]).reshape(4, 4)

print("\nLearned Policy:")
print(policy_grid)


# Success Rate
success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward (last 1000 episodes):", success_rate)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500
moving_avg = np.convolve(
    episode_rewards,
    np.ones(window)/window,
    mode='valid'
)

plt.plot(moving_avg)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Learning Curve")
plt.grid()
plt.show()


env.close()

```

---

## Output


<img width="980" height="602" alt="image" src="https://github.com/user-attachments/assets/21431ce9-af7a-44dd-b799-df13733a5ada" />


## Average reward over last 2000 episodes:

<img width="720" height="475" alt="image" src="https://github.com/user-attachments/assets/7a526cb3-6d0d-432f-8f42-edb00575fc25" />

---

## Result
```text

Thus, the On-Policy Monte Carlo Control algorithm was successfully
implemented using the Gymnasium FrozenLake-v1 environment.

The agent successfully estimated the action-value function Q(s,a)
from complete episodes and learned an improved policy using the
epsilon-greedy strategy.

```
---

## Inference
```text
The experiment demonstrates that Monte Carlo Control can learn an
effective policy without prior knowledge of the environment model.

By generating complete episodes and updating the Q-values using
the observed returns, the agent gradually improves its decision-making
ability and learns to reach the goal while avoiding holes in the
FrozenLake environment.


```





---

