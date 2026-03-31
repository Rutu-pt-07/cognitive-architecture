# Q-Learning Agent — Code Explanation

## What is this file?

This is the **"learning" agent** — the opposite of the Production Agent. Instead of following fixed rules, it starts knowing **nothing** and learns the best moves through trial and error. It uses a classic Reinforcement Learning algorithm called **Q-Learning**.

---

## The Core Idea: The Q-Table

Imagine a giant cheat-sheet (a 3D table) with one entry for every possible `(row, col, action)` combination. For a 5×5 grid with 4 actions, that's `5 × 5 × 4 = 100` entries.

Each entry answers the question: **"If I'm at cell (r, c) and I take action X, how much total future reward can I expect?"**

- At the start, every entry is `0` (the agent knows nothing).
- As it plays episodes, it updates these numbers based on what actually happens.
- After enough training, the table becomes a perfect map: at any cell, just pick the action with the highest Q-value.

---

## How it works, step by step:

### The `__init__` method (Line 25-45)
Sets up the agent with:
- **`alpha` (0.1)** — Learning rate. How much new information overrides old. Think of it as "how fast do I update my beliefs." Too high = unstable, too low = learns too slowly.
- **`gamma` (0.95)** — Discount factor. How much the agent cares about future rewards vs immediate rewards. 0.95 means it strongly values long-term payoff.
- **`epsilon` (starts at 1.0)** — Exploration rate. At 1.0, the agent picks random moves 100% of the time (pure exploration). This decays over time to 0.01, so eventually it almost always picks the best known move (exploitation).
- **`epsilon_decay` (0.995)** — After each episode, epsilon is multiplied by this. 1.0 → 0.995 → 0.990 → ... slowly shifting from exploration to exploitation.
- **`q_table`** — The cheat-sheet. Initialized to all zeros using numpy.

### The `reset_q_table` method (Line 47-54)
Wipes the Q-table clean and resets epsilon back to 1.0. Used when the agent needs to "forget everything" and start learning from scratch (like when the environment changes).

---

## The Decision: the `act` method (Line 56-72)

This is the **ε-greedy policy**:
1. Generate a random number between 0 and 1.
2. If it's less than `epsilon` → pick a **random action** (explore — try something new).
3. Otherwise → pick the action with the **highest Q-value** for the current state (exploit — use what you know).

**Why explore?** Without exploration, the agent might find one okay path and never discover a better one. By occasionally trying random moves, it can stumble onto shortcuts it would never have found by being greedy.

**Example**: epsilon = 0.3 means 30% of the time the agent explores randomly, 70% it uses its best known action.

---

## The Learning: the `update` method (Line 74-84)

This is the **Bellman equation** — the mathematical heart of Q-Learning:

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s', ·) − Q(s, a) ]
```

In plain English:
1. **`td_target`** = The reward I just got (`r`) + the discounted value of the best thing I can do from the next state (`γ × max Q(s', ·)`). This is what I *should* have expected.
2. **`td_error`** = `td_target - Q(s, a)`. This is the surprise — the difference between what happened and what I predicted.
3. **Update** = Nudge the old Q-value toward the target by a small step (`α × td_error`).

**Example walkthrough**:
- Agent is at `(1, 2)`, takes action RIGHT, lands at `(1, 3)`, gets reward `-1` (normal step).
- Old Q-value for `(1, 2, RIGHT)` was `2.0`.
- Best Q-value at `(1, 3)` across all actions is `5.0`.
- `td_target = -1 + 0.95 × 5.0 = 3.75`
- `td_error = 3.75 - 2.0 = 1.75`
- New Q = `2.0 + 0.1 × 1.75 = 2.175`

The Q-value went up slightly — the agent now knows this move is a bit better than it thought.

### The `decay_epsilon` method (Line 86-89)
After each episode, multiply epsilon by 0.995. This gradually shifts the agent from "trying random stuff" (early training) to "using what I've learned" (late training).

---

## The Training Loop: the `train` method (Line 91-131)

This runs the full learning process over many episodes (default: 1000):

For each episode:
1. Reset the environment (agent goes back to start).
2. Loop until the episode ends (goal reached or timeout):
   - Pick an action using `act()` (ε-greedy).
   - Take the step in the environment, get reward.
   - Update the Q-table using `update()`.
   - Accumulate total reward.
3. Decay epsilon (explore less next time).
4. Record the episode's total reward and step count in history lists.
5. Every 200 episodes, print a progress update showing the average reward of the last 50 episodes.

Returns the full reward and step history — used later for plotting learning curves.

---

## The Evaluation: the `run_episode` method (Line 133-157)

Runs one episode **without learning** — just to see how well the agent performs with its current Q-table:
- `explore=False` by default → always picks the greedy best action (no random moves).
- Doesn't call `update()` → the Q-table stays unchanged.
- Returns: total reward, steps, success (bool), and the path walked.

This is used after training to measure the agent's final performance.

---

## Save/Load: `save_q_table` and `load_q_table` (Line 159-165)

Simple utility to save the learned Q-table to a `.npy` file and load it back. This means you can train once, save, and reuse the knowledge later without retraining.

---

## The Key Takeaway

This agent starts as a **blank slate** and builds up knowledge through experience. Early on, it's terrible — bumbling around randomly. But after hundreds of episodes, the Q-table converges to near-optimal values, and the agent navigates efficiently.

The critical advantage over the Production Agent: **when the environment changes, you can retrain this agent** and it will learn the new layout. The Production Agent is stuck with its hardcoded rules forever.

### The Learning Journey (what the training curves show):
1. **Episodes 1-200**: Mostly random, low rewards, lots of timeouts
2. **Episodes 200-500**: Starting to learn good moves, rewards improving
3. **Episodes 500-1000**: Converged — consistently reaching the goal in minimum steps
