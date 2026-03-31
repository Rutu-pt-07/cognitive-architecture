# Production Agent — Code Explanation

## What is this file?

This is the **"dumb but reliable" agent** — it doesn't learn anything. It uses a fixed set of IF-THEN rules (called "production rules," inspired by the SOAR cognitive architecture) to decide which direction to move at every step.

---

## How it works, step by step:

### The `__init__` method (Line 26-28)
When you create the agent, it just sets its name and creates an empty list called `rule_log`. This log records which rule fired at each step — useful for debugging ("why did it go LEFT there?").

### The `reset` method (Line 30-32)
Clears the rule log at the start of each new episode. The agent has no memory between episodes — it starts fresh every time.

### The `_is_safe` helper (Line 34-38)
A simple check: "Is this cell inside the grid AND not an obstacle?" Returns `True` or `False`. This is used by the rules below to avoid walking into walls or obstacles.

### The `_manhattan_distance` helper (Line 40-42)
Calculates the "city block distance" between two cells — how many steps you'd need if you could walk in a straight line with no obstacles. For example, from `(0,0)` to `(4,4)` = `|0-4| + |0-4|` = 8. This helps the agent judge which moves bring it closer to the goal.

---

## The Brain: the `act` method (Line 44-107)

This is the core decision-making. It runs **three rules in priority order** — the first one that works gets used:

### Rule 1: "Move directly toward the goal" (Line 62-81)
- It checks: is the goal below me? Then consider DOWN. Is it to my right? Then consider RIGHT. Etc.
- For each goal-facing direction, it checks if that cell is safe.
- If yes → take that action immediately. This is the greedy, direct approach.
- **Example**: Agent is at `(1,2)`, goal is at `(4,4)`. Goal is below (row 4 > row 1) → try DOWN. Goal is to the right (col 4 > col 2) → try RIGHT. Whichever is safe first, it takes.

### Rule 2: "Pick the best safe alternative" (Line 83-99)
- If Rule 1 failed (all goal-facing moves are blocked by obstacles), it looks at ALL four directions.
- For each safe move, it calculates the Manhattan distance from that new cell to the goal.
- It picks the move that gets it **closest to the goal**, even if it's not directly toward it.
- **Example**: Can't go DOWN or RIGHT because of obstacles. Going UP from `(2,2)` gives distance 8, going LEFT gives distance 9. It picks UP.

### Rule 3: "Random panic move" (Line 101-107)
- If even Rule 2 found nothing (all 4 neighbors are walls or obstacles — very rare), it just picks a random direction and hopes for the best.

---

## The `run_episode` method (Line 109-135)

This runs one complete game:
1. Reset the agent's log
2. Place the agent at the start position
3. Loop: ask `act()` for a move → take the step in the environment → accumulate reward → repeat until done (goal reached or timeout)
4. Return a dictionary with: total reward earned, number of steps taken, whether it succeeded, the path it walked, and which rules fired

---

## The Key Takeaway

This agent is **hard-coded intelligence**. It always does the same thing in the same situation. It can't adapt — if you rearrange the obstacles in a way that its greedy "move toward goal" strategy leads into a dead end, it will fail repeatedly with no way to improve. That's the whole point of comparing it against the Q-Learning agent, which CAN learn from mistakes.
