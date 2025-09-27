import pygame
import random
import sys

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 400, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 32)

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 206, 235)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

# Bird
BIRD_X, BIRD_Y = 60, HEIGHT // 2
BIRD_WIDTH, BIRD_HEIGHT = 34, 24
GRAVITY = 0.5
FLAP_POWER = -8

# Pipes
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 150
PIPE_SPEED = 3

def draw_bird(y):
    pygame.draw.rect(SCREEN, RED, (BIRD_X, y, BIRD_WIDTH, BIRD_HEIGHT))

def draw_pipe(x, top_height):
    pygame.draw.rect(SCREEN, GREEN, (x, 0, PIPE_WIDTH, top_height))
    pygame.draw.rect(SCREEN, GREEN, (x, top_height + PIPE_GAP, PIPE_WIDTH, HEIGHT - top_height - PIPE_GAP))

def main():
    bird_y = BIRD_Y
    bird_vel = 0
    pipes = []
    score = 0
    running = True

    # Create first pipe
    pipe_x = WIDTH
    pipe_height = random.randint(50, HEIGHT - PIPE_GAP - 50)
    pipes.append([pipe_x, pipe_height])

    while running:
        CLOCK.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_vel = FLAP_POWER

        # Bird physics
        bird_vel += GRAVITY
        bird_y += bird_vel

        # Pipes logic
        for pipe in pipes:
            pipe[0] -= PIPE_SPEED

        # Add new pipes
        if pipes[-1][0] < WIDTH - 200:
            pipe_x = WIDTH
            pipe_height = random.randint(50, HEIGHT - PIPE_GAP - 50)
            pipes.append([pipe_x, pipe_height])

        # Remove off-screen pipes
        if pipes[0][0] < -PIPE_WIDTH:
            pipes.pop(0)
            score += 1

        # Collision
        for pipe in pipes:
            if (BIRD_X + BIRD_WIDTH > pipe[0] and BIRD_X < pipe[0] + PIPE_WIDTH):
                if (bird_y < pipe[1]) or (bird_y + BIRD_HEIGHT > pipe[1] + PIPE_GAP):
                    running = False
        if bird_y > HEIGHT or bird_y < 0:
            running = False

        # Drawing
        SCREEN.fill(BLUE)
        draw_bird(bird_y)
        for pipe in pipes:
            draw_pipe(pipe[0], pipe[1])
        score_surface = FONT.render(f"Score: {score}", True, WHITE)
        SCREEN.blit(score_surface, (10, 10))
        pygame.display.flip()

    # Game over screen
    SCREEN.fill(BLUE)
    over_surface = FONT.render("Game Over!", True, RED)
    SCREEN.blit(over_surface, (WIDTH // 2 - over_surface.get_width() // 2, HEIGHT // 2 - 40))
    score_surface = FONT.render(f"Score: {score}", True, WHITE)
    SCREEN.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, HEIGHT // 2 + 10))
    pygame.display.flip()
    pygame.time.wait(2000)

if __name__ == "__main__":
    main()
