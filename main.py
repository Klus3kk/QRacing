import pygame
import sys
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Force TensorFlow to use CPU to avoid CUDA errors
tf.config.set_visible_devices([], "GPU")

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
ACCELERATION = 0.2
FRICTION = 0.05
ROTATE_SPEED = 5  # Better turning behavior
SENSOR_LENGTH = 120
NUM_SENSORS = 5
SENSOR_ANGLES = [-75, -45, 0, 45, 75]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Racing Car - Deep Q-Learning")

rows, cols = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

# Generate Random Maze
def generate_maze():
    maze = np.ones((rows, cols), dtype=int)

    def carve_maze(x, y):
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < rows and 0 < ny < cols and maze[nx, ny] == 1:
                maze[x + dx // 2, y + dy // 2] = 0
                maze[nx, ny] = 0
                carve_maze(nx, ny)

    maze[1, 1] = 0
    carve_maze(1, 1)

    # Set random finish line (ensures it's not the start position)
    finish_x, finish_y = rows - 2, cols - 2
    maze[finish_x, finish_y] = 2  # Mark the finish line
    return maze, (finish_x, finish_y)

# Define actions
ACTIONS = {
    0: "accelerate",
    1: "brake",
    2: "turn_left",
    3: "turn_right",
    4: "no_action"
}

# Deep Q-Network parameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000
UPDATE_TARGET_EVERY = 50

# Replay memory
memory = deque(maxlen=MEMORY_SIZE)

# Build Deep Q-Network
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(9,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(ACTIONS), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# Get state representation
def get_state(player_x, player_y, player_angle, player_velocity, sensors):
    return np.array([
        player_x / WIDTH, player_y / HEIGHT,
        player_angle / 360, player_velocity / 10,
        *[s / SENSOR_LENGTH for s in sensors]
    ])

# Choose action using ε-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < EXPLORATION_RATE:
        return random.choice(list(ACTIONS.keys()))
    q_values = model.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])

# Reward function
def get_reward(collision, finished, velocity, out_of_bounds, rotated):
    if finished:
        return 1500  # Big reward for finishing (promotes goal-seeking)
    if out_of_bounds:
        return -500  # Major penalty for leaving the map
    if collision:
        return -100  # Hitting a wall
    if rotated:
        return 50  # Reward for turning (encourages smarter navigation)
    return velocity * 0.5  # Encourages movement

# Sensor function
def get_sensors(player_x, player_y, player_angle, maze):
    sensor_readings = []
    for angle_offset in SENSOR_ANGLES:
        sensor_angle = math.radians(player_angle + angle_offset)
        for length in range(0, SENSOR_LENGTH, 5):
            sensor_x = int(player_x + length * math.cos(sensor_angle))
            sensor_y = int(player_y - length * math.sin(sensor_angle))
            if 0 <= sensor_x < WIDTH and 0 <= sensor_y < HEIGHT:
                if maze[int(sensor_x // GRID_SIZE), int(sensor_y // GRID_SIZE)] == 1:
                    sensor_readings.append(length)
                    break
        else:
            sensor_readings.append(SENSOR_LENGTH)
    return sensor_readings

# Draw the maze
def draw_maze(maze, finish_line):
    for x in range(rows):
        for y in range(cols):
            if maze[x, y] == 1:
                pygame.draw.rect(screen, BLACK, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            elif (x, y) == finish_line:
                pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Finish line

# Draw the car as a triangle
def draw_car(player_x, player_y, player_angle):
    car_length = 10
    car_width = 6

    angle_rad = math.radians(player_angle)
    front = (player_x + car_length * math.cos(angle_rad), player_y - car_length * math.sin(angle_rad))
    left = (player_x + car_width * math.cos(angle_rad + math.pi * 2 / 3), player_y - car_width * math.sin(angle_rad + math.pi * 2 / 3))
    right = (player_x + car_width * math.cos(angle_rad - math.pi * 2 / 3), player_y - car_width * math.sin(angle_rad - math.pi * 2 / 3))

    pygame.draw.polygon(screen, BLUE, [front, left, right])

# Main game loop
def main():
    global EXPLORATION_RATE
    clock = pygame.time.Clock()
    maze, finish_line = generate_maze()  # Initial maze generation

    for episode in range(5000):
        player_x, player_y, player_angle, player_velocity = 20, 20, 90, 0
        done = False
        sensors = get_sensors(player_x, player_y, player_angle, maze)
        state = get_state(player_x, player_y, player_angle, player_velocity, sensors)

        while not done:
            action = choose_action(state)

            rotated = False
            if action == 0:
                player_velocity += ACCELERATION
            elif action == 1:
                player_velocity -= ACCELERATION
            elif action == 2:
                player_angle += ROTATE_SPEED
                rotated = True
            elif action == 3:
                player_angle -= ROTATE_SPEED
                rotated = True

            new_x = player_x + math.cos(math.radians(player_angle)) * player_velocity
            new_y = player_y - math.sin(math.radians(player_angle)) * player_velocity

            out_of_bounds = new_x < 0 or new_x >= WIDTH or new_y < 0 or new_y >= HEIGHT
            collision = not out_of_bounds and maze[int(new_x // GRID_SIZE), int(new_y // GRID_SIZE)] == 1
            finished = not out_of_bounds and (int(new_x // GRID_SIZE), int(new_y // GRID_SIZE)) == finish_line

            if finished:
                maze, finish_line = generate_maze()  # Generate a new maze
                break  # Restart game with new maze

            if collision or out_of_bounds:
                done = True  # Reset only on collision
            else:
                player_x, player_y = new_x, new_y

            screen.fill(WHITE)
            draw_maze(maze, finish_line)  # Draw the maze and finish line
            draw_car(player_x, player_y, player_angle)
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    main()
