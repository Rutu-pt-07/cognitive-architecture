"""
Q-Learning Agent v2 — Tabular Q-learning for 10x10 grid.
"""
import numpy as np

ALL_ACTIONS = [0, 1, 2, 3]


class QLearningAgentV2:
    def __init__(self, grid_size, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.name = "Q-Learning Agent"
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((grid_size, grid_size, len(ALL_ACTIONS)))
        self.rewards_history = []
        self.steps_history = []

    def reset_q_table(self):
        self.q_table = np.zeros((self.grid_size, self.grid_size, len(ALL_ACTIONS)))
        self.epsilon = 1.0
        self.rewards_history = []
        self.steps_history = []

    def act(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(ALL_ACTIONS)
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state, done):
        r, c = state
        nr, nc = next_state
        best_next = np.max(self.q_table[nr, nc]) if not done else 0
        td_target = reward + self.gamma * best_next
        self.q_table[r, c, action] += self.alpha * (td_target - self.q_table[r, c, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes, verbose_every=200):
        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0
            while not env.done:
                action = self.act(state, explore=True)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            self.decay_epsilon()
            self.rewards_history.append(total_reward)
            self.steps_history.append(env.steps)
            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(self.rewards_history[-50:])
                print(f"    Ep {ep+1}/{num_episodes} | Avg(50)={avg:.1f} | eps={self.epsilon:.4f}")
        return {"rewards_history": list(self.rewards_history),
                "steps_history": list(self.steps_history)}

    def run_episode(self, env, explore=False):
        state = env.reset()
        total_reward = 0
        path = [state]
        while not env.done:
            action = self.act(state, explore=explore)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            path.append(state)
        return {"total_reward": total_reward, "steps": env.steps,
                "success": state == env.goal, "path": path}
