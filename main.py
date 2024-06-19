import pygame
import sys
import math
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
ACCELERATION = 0.2
FRICTION = 0.05
ROTATE_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Setup the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Racing Game")

# Load assets
player_car = pygame.image.load('Race_Car.png').convert_alpha()
map1 = pygame.image.load('Map1.png').convert_alpha()

# Resize
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

# Main game loop
def main():
    clock = pygame.time.Clock()
    running = True
    # Debug mode 
    debug_mode = False
    debug_pressed = False
    
    acc_pressed = False
    acc_used = False
    # Text
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    
    # Store the start time
    start_time = time.time()
    
    # Player variables
    player_x = 200
    player_y = 500
    player_angle = 90
    player_velocity = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not event.type == pygame.KEYDOWN:
                if debug_pressed:
                    debug_pressed = False  
                if acc_pressed:
                    acc_pressed = False                                    
        
        # Handle keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_angle += ROTATE_SPEED
        if keys[pygame.K_RIGHT]:
            player_angle -= ROTATE_SPEED
        if keys[pygame.K_UP]:
            player_velocity -= ACCELERATION
        if keys[pygame.K_DOWN]:
            player_velocity += ACCELERATION
        if keys[pygame.K_LSHIFT] and not acc_pressed and not acc_used:
            player_velocity = -15  
            acc_pressed = True      
            acc_used = True    

        if keys[pygame.K_d] and not debug_pressed:
            debug_mode = not debug_mode
            debug_pressed = True          
        if keys[pygame.K_e]:
            running = False

        # Apply friction
        player_velocity *= (1 - FRICTION)
        
        player_dx = math.cos(math.radians(player_angle)) * player_velocity
        player_dy = math.sin(math.radians(player_angle)) * player_velocity
        
        player_x -= player_dx
        player_y += player_dy
        
        # Update player's rectangle position 
        player_rect.center = (player_x, player_y)

        if player_x < 0:
            player_x = 0
        if player_x > WIDTH:
            player_x = WIDTH
        if player_y < 0:
            player_y = 0
        if player_y > HEIGHT:
            player_y = HEIGHT

        # Render
        screen.fill(WHITE)  # Clear screen with white background
        
        # Draw the map
        screen.blit(map1, map1_rect.topleft)

        # Draw the finish line
        pygame.draw.rect(screen, RED, finish_line)
        
        # Rotate the car image
        rotated_car = pygame.transform.rotate(player_car, player_angle)
        rotated_rect = rotated_car.get_rect(center=(player_x, player_y))
        
        # Create a mask for the rotated car
        rotated_car_mask = pygame.mask.from_surface(rotated_car)
        
        # Calculate the offset between the car and the map
        offset = (rotated_rect.left - map1_rect.left, rotated_rect.top - map1_rect.top)
        
        # Check for collision
        collision_point = map_mask.overlap(rotated_car_mask, offset)
        if collision_point:
            running = False
            
        if rotated_rect.colliderect(finish_line):
            running = False
                        
        screen.blit(rotated_car, rotated_rect.topleft)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        time_str = f"Time: {elapsed_time:.2f}s"
        text = time_str
        if debug_mode:
            text = time_str + " " + str(round(player_x)) + " " + str(round(player_y))            
        # Render the text
        text_surface = my_font.render(text, True, BLACK)
        screen.blit(text_surface, (0, 0))

        pygame.display.flip()  # Update the full display surface to the screen
        
        # Cap the frame rate
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
