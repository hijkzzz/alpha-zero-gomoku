# -*- coding: utf-8 -*-
import pygame
import os

class BoardGUI():
    def __init__(self, n=15, nir=5):
        
        # color
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)

        # resolution
        self.WIDTH = 720
        self.HEIGHT = 720
        self.GRID_WIDTH = self.WIDTH // 20


        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Gomoku")


        # timer
        self.FPS = 30
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(os.path.join(base_folder, 'back.png')).convert()

    def loop(self):
        running = True
        while running:
            # timer
            self.clock.tick(self.FPS)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    running = False

            # draw
            self.draw_background(self.screen)

            # refresh 
            pygame.display.flip()


    # 画出棋盘
    def draw_background(self, surf):
        # load background
        surf.blit(self.background_img, (0, 0))

        # draw lines
        rect_lines = [
            ((self.GRID_WIDTH, self.GRID_WIDTH), (self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.GRID_WIDTH), (self.WIDTH - self.GRID_WIDTH, self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH),
                (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.WIDTH - self.GRID_WIDTH, self.GRID_WIDTH),
                (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
        ]
        for line in rect_lines:
            pygame.draw.line(surf, self.BLACK, line[0], line[1], 2)

        # draw grid
        for i in range(17):
            pygame.draw.line(surf, self.BLACK,
                            (self.GRID_WIDTH * (2 + i), self.GRID_WIDTH),
                            (self.GRID_WIDTH * (2 + i), self.HEIGHT - self.GRID_WIDTH))
            pygame.draw.line(surf, self.BLACK,
                            (self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)),
                            (self.HEIGHT - self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)))

        # draw chessmen
        circle_center = [
            (self.GRID_WIDTH * 4, self.GRID_WIDTH * 4),
            (self.WIDTH - self.GRID_WIDTH * 4, self.GRID_WIDTH * 4),
            (self.WIDTH - self.GRID_WIDTH * 4, self.HEIGHT - self.GRID_WIDTH * 4),
            (self.GRID_WIDTH * 4, self.HEIGHT - self.GRID_WIDTH * 4),
            (self.GRID_WIDTH * 10, self.GRID_WIDTH * 10)
        ]
        for cc in circle_center:
            pygame.draw.circle(surf, self.BLACK, cc, 15)

# test
if __name__ == "__main__":
    board_gui = BoardGUI()
    board_gui.loop()