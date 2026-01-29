import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# =========================
# PARAMETERS
# =========================
IMAGE_PATH = r"directory"   # <-- CHANGE THIS

EPISODES = 4000
MAX_STEPS = 2000

ALPHA = 0.1          # learning rate
GAMMA = 0.95         # discount factor
EPSILON = 0.2        # exploration rate

# Rewards
REWARD_FREE = -0.05
REWARD_GREEN = -1.0
REWARD_BLUE = 50.0
REWARD_RED = -100.0

# Color thresholds (BGR)
RED_MIN   = np.array([0, 0, 150])
RED_MAX   = np.array([80, 80, 255])

GREEN_MIN = np.array([0, 150, 0])
GREEN_MAX = np.array([80, 255, 80])

BLUE_MIN  = np.array([150, 0, 0])
BLUE_MAX  = np.array([255, 80, 80])


# =========================
# ROBUST IMAGE LOADING
# =========================
def load_color_image(path):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# =========================
# BUILD ENVIRONMENT
# =========================
def build_environment(image_bgr):
    h, w, _ = image_bgr.shape

    reward_map = np.full((h, w), REWARD_FREE, dtype=float)
    forbidden = np.zeros((h, w), dtype=bool)

    red_mask = cv2.inRange(image_bgr, RED_MIN, RED_MAX)
    green_mask = cv2.inRange(image_bgr, GREEN_MIN, GREEN_MAX)
    blue_mask = cv2.inRange(image_bgr, BLUE_MIN, BLUE_MAX)

    reward_map[green_mask > 0] = REWARD_GREEN
    reward_map[blue_mask > 0] = REWARD_BLUE
    reward_map[red_mask > 0] = REWARD_RED
    forbidden[red_mask > 0] = True

    blue_positions = np.argwhere(blue_mask > 0)

    return reward_map, forbidden, blue_positions


# =========================
# Q-LEARNING
# =========================
def train_q_learning(reward_map, forbidden, start, goal):
    h, w = reward_map.shape
    q_table = np.zeros((h, w, 4))

    actions = [(-1,0), (1,0), (0,-1), (0,1)]

    for episode in range(EPISODES):
        x, y = start

        for _ in range(MAX_STEPS):

            # ε-greedy policy
            if random.random() < EPSILON:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[x, y])

            dx, dy = actions[action]
            nx, ny = x + dx, y + dy

            # Boundary or forbidden
            if nx < 0 or ny < 0 or nx >= h or ny >= w or forbidden[nx, ny]:
                reward = REWARD_RED
                q_table[x, y, action] += ALPHA * (reward - q_table[x, y, action])
                break

            reward = reward_map[nx, ny]

            q_table[x, y, action] += ALPHA * (
                reward + GAMMA * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            x, y = nx, ny

            if (x, y) == goal:
                break

        if episode % 500 == 0:
            print(f"Episode {episode}")

    return q_table


# =========================
# EXTRACT POLICY
# =========================
def extract_path(q_table, start, goal):
    path = []
    x, y = start
    actions = [(-1,0), (1,0), (0,-1), (0,1)]

    for _ in range(MAX_STEPS):
        path.append((x, y))
        action = np.argmax(q_table[x, y])
        dx, dy = actions[action]
        x, y = x + dx, y + dy

        if (x, y) == goal:
            path.append(goal)
            return path

    return path


# =========================
# MAIN PIPELINE
# =========================
def main():
    image_bgr = load_color_image(IMAGE_PATH)
    if image_bgr is None:
        raise ValueError("Could not load image")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    start = (h - 1, w // 2)

    reward_map, forbidden, blue_positions = build_environment(image_bgr)
    goal = tuple(random.choice(blue_positions))

    print("Training Q-learning agent...")
    q_table = train_q_learning(reward_map, forbidden, start, goal)

    path = extract_path(q_table, start, goal)

    # =========================
    # VISUALIZATION
    # =========================
    output = image.copy()

    for x, y in path:
        output[x, y] = [180, 0, 180]  # purple path

    output[start] = [255, 255, 0]    # start
    output[goal] = [0, 255, 255]     # goal

    plt.figure(figsize=(7, 7))
    plt.imshow(output)
    plt.title("Q-learning path to blue rock")
    plt.axis("off")
    plt.show()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
