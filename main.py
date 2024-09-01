import pygame
import sys
import math
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
ACCELERATION = 0.2
FRICTION = 0.05
ROTATE_SPEED = 5
SENSOR_LENGTH = 100  # Length of the sensors

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Setup the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Racing Game with Q-learning")

# Load assets with error handling
try:
    player_car = pygame.image.load('Race_Car.png').convert_alpha()
    map1 = pygame.image.load('Map1.png').convert_alpha()
except pygame.error as e:
    print(f"Failed to load assets: {e}")
    pygame.quit()
    sys.exit()

# Resize assets
CAR_WIDTH, CAR_HEIGHT = 40, 20
player_car = pygame.transform.scale(player_car, (CAR_WIDTH, CAR_HEIGHT))
MAP_WIDTH, MAP_HEIGHT = 500, 500
map1 = pygame.transform.scale(map1, (MAP_WIDTH, MAP_HEIGHT))

# Create masks for pixel-perfect collision detection
car_mask = pygame.mask.from_surface(player_car)
map_mask = pygame.mask.from_surface(map1)

# Create game objects
player_rect = player_car.get_rect(center=(WIDTH // 2, HEIGHT // 2))
map1_rect = map1.get_rect(center=(WIDTH // 2, HEIGHT // 2))

# Create finish line
finish_line = pygame.Rect(561, 68, 82, 40)

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# Initialize Q-table
states = (WIDTH // 10, HEIGHT // 10, 36, 20)
q_table = np.zeros(states + (5,))  # 5 possible actions

# Define actions
ACTIONS = {
    0: "accelerate",
    1: "turn_left",
    2: "turn_right",
}

def get_state(player_x, player_y, player_angle, player_velocity):
    x_state = int(player_x // 10)
    y_state = int(player_y // 10)
    angle_state = int(player_angle // 10) % 36  # Wrap around at 360 degrees
    velocity_state = int(player_velocity // 1) % 20  # Cap the velocity state
    return (x_state, y_state, angle_state, velocity_state)

def choose_action(state):
    if random.uniform(0, 1) < EXPLORATION_RATE:
        return random.choice(list(ACTIONS.keys()))
    else:
        return np.argmax(q_table[state])

def get_reward(collision, finished, player_velocity):
    if collision:
        return -100
    elif finished:
        return 100
    else:
        return player_velocity * 0.1

def reset_game():
    player_x = 200
    player_y = 500
    player_angle = 90
    player_velocity = 0
    return player_x, player_y, player_angle, player_velocity

def check_sensor(sensor_x, sensor_y):
    # Check if the sensor detects a collision (black pixel)
    if 0 <= sensor_x < MAP_WIDTH and 0 <= sensor_y < MAP_HEIGHT:
        return map1.get_at((int(sensor_x), int(sensor_y))) == BLACK
    return False

def get_sensors(player_x, player_y, player_angle):
    sensors = []
    angles = [-45, 0, 45]  # Sensors at -45, 0, 45 degrees relative to car angle

    for angle_offset in angles:
        sensor_angle = math.radians(player_angle + angle_offset)
        sensor_x = player_x + SENSOR_LENGTH * math.cos(sensor_angle)
        sensor_y = player_y - SENSOR_LENGTH * math.sin(sensor_angle)
        collision = check_sensor(sensor_x, sensor_y)
        sensors.append(((sensor_x, sensor_y), collision))

    return sensors

def draw_sensors(sensors):
    for sensor, collision in sensors:
        color = RED if collision else GREEN
        pygame.draw.line(screen, color, player_rect.center, sensor, 2)

def main():
    global EXPLORATION_RATE
    clock = pygame.time.Clock()
    
    total_episodes = 10000
    for episode in range(total_episodes):
        player_x, player_y, player_angle, player_velocity = reset_game()
        done = False
        
        state = get_state(player_x, player_y, player_angle, player_velocity)
        episode_reward = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            action = choose_action(state)
            
            if action == 0:  # accelerate
                player_velocity += ACCELERATION
            elif action == 1:  # brake
                player_velocity -= ACCELERATION
            elif action == 2:  # turn left
                player_angle += ROTATE_SPEED
            elif action == 3:  # turn right
                player_angle -= ROTATE_SPEED
            
            player_velocity *= (1 - FRICTION)
            
            player_dx = math.cos(math.radians(player_angle)) * player_velocity
            player_dy = math.sin(math.radians(player_angle)) * player_velocity
            
            player_x += player_dx
            player_y -= player_dy
            
            if player_x < 0:
                player_x = 0
            if player_x > WIDTH:
                player_x = WIDTH
            if player_y < 0:
                player_y = 0
            if player_y > HEIGHT:
                player_y = HEIGHT

            rotated_car = pygame.transform.rotate(player_car, player_angle)
            rotated_rect = rotated_car.get_rect(center=(player_x, player_y))
            rotated_car_mask = pygame.mask.from_surface(rotated_car)
            offset = (rotated_rect.left - map1_rect.left, rotated_rect.top - map1_rect.top)
            
            try:
                collision_point = map_mask.overlap(rotated_car_mask, offset)
            except Exception as e:
                print(f"Error calculating collision: {e}")
                collision_point = None
            
            finished = rotated_rect.colliderect(finish_line)
            collision = bool(collision_point)
            
            reward = get_reward(collision, finished, player_velocity)
            episode_reward += reward
            
            next_state = get_state(player_x, player_y, player_angle, player_velocity)
            
            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
            q_table[state + (action,)] = new_value
            
            state = next_state
            
            if collision or finished:
                done = True

            # Get and draw sensors
            sensors = get_sensors(player_x, player_y, player_angle)
            
            screen.fill(WHITE)
            screen.blit(map1, map1_rect.topleft)
            pygame.draw.rect(screen, RED, finish_line)
            screen.blit(rotated_car, rotated_rect.topleft)
            draw_sensors(sensors)
            pygame.display.flip()
            
            clock.tick(FPS)
        
        EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {episode_reward:.2f}, Exploration Rate: {EXPLORATION_RATE:.4f}")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
