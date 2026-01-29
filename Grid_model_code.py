import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# ======================
# PARAMETERS
# ======================
GRID_SIZE = 30
OBSTACLE_PROBABILITY = 0.3

iterationS = 3000
CHECKPOINTS = [1000, 2000, 3000]

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.2

MAX_STEPS_PER_iteration = 800
ANIMATION_DELAY = 0.15


# ======================
# GRID GENERATION
# ======================
def generate_grid(size=30, obstacle_prob=0.3):
    """
    Cell values:
      0   -> normal cell
     -1   -> low penalty (easy to cross)
     -3   -> medium penalty
     -5   -> high penalty (dangerous)
    """
    grid = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i, j] = random.choice([-1, -3, -5])

    start = (0, 0)
    goal = (size - 1, size - 1)

    grid[start] = 0
    grid[goal] = 0

    return grid, start, goal


# ======================
# ENVIRONMENT
# ======================
class GridWorld:
    def __init__(self, size=30):
        self.size = size
        self.grid, self.start, self.goal = generate_grid(size, OBSTACLE_PROBABILITY)
        self.agent_position = self.start

    def reset(self):
        """Reset agent position to the start cell"""
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        """
        Actions:
        0 -> up
        1 -> down
        2 -> left
        3 -> right
        """
        x, y = self.agent_position

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        # Out of bounds
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return self.agent_position, -1, False

        self.agent_position = (x, y)

        # Goal reached
        if self.agent_position == self.goal:
            return self.agent_position, 20, True

        # Cell penalty (if any)
        cell_reward = self.grid[x, y]
        if cell_reward < 0:
            return self.agent_position, cell_reward, False

        # Normal step cost
        return self.agent_position, -0.05, False


# ======================
# VISUALIZATION UTILITIES
# ======================
def get_cell_color(value):
    """Return RGB color depending on cell penalty"""
    if value == -5:
        return [1, 0, 0]        # red (high danger)
    if value == -3:
        return [1, 0.5, 0]      # orange (medium danger)
    if value == -1:
        return [1, 0.7, 0.8]    # pink (low danger)
    return [1, 1, 1]            # white (normal)


def demonstrate_policy(env, q_table, title):
    """Animate the agent following a learned policy"""
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    state = env.reset()
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_iteration:
        ax.clear()

        display_grid = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                display_grid[i, j] = get_cell_color(env.grid[i, j])

        # Start and goal
        sx, sy = env.start
        gx, gy = env.goal
        display_grid[sx, sy] = [0, 0, 1]   # blue
        display_grid[gx, gy] = [0, 1, 0]   # green

        # Agent
        x, y = state
        display_grid[x, y] = [1, 1, 0]     # yellow

        ax.imshow(display_grid)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

        plt.pause(ANIMATION_DELAY)

        action = np.argmax(q_table[x, y])
        state, reward, done = env.step(action)
        steps += 1

    plt.ioff()
    plt.show()


# ======================
# TRAINING (Q-LEARNING)
# ======================
training_env = GridWorld(GRID_SIZE)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
snapshots = {}

for iteration in range(1, iterationS + 1):
    state = training_env.reset()
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_iteration:
        x, y = state

        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[x, y])

        next_state, reward, done = training_env.step(action)
        nx, ny = next_state

        # Q-learning update
        q_table[x, y, action] += LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )

        state = next_state
        steps += 1

    # Save snapshots for visualization
    if iteration in CHECKPOINTS:
        snapshots[iteration] = copy.deepcopy(q_table)
        print(f"Saved snapshot at iteration {iteration}")


# ======================
# DEMONSTRATIONS
# ======================
for iteration in CHECKPOINTS:
    demo_env = GridWorld(GRID_SIZE)

    # Use the SAME grid as training for fair comparison
    demo_env.grid = training_env.grid
    demo_env.start = training_env.start
    demo_env.goal = training_env.goal

    demonstrate_policy(
        demo_env,
        snapshots[iteration],
        title=f"Policy after {iteration} iterations (variable-cost grid)"
    )
