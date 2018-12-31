# -*- coding: utf-8 -*-
import pygame
import os
import numpy as np

class BoardGUI():
    def __init__(self, board=None, human=False, fps=60):

        # color
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)

        # resolution
        self.WIDTH = 600
        self.HEIGHT = 600
        
        self.board = None
        self.n = None
        self.GRID_WIDTH = None

        # other
        self.FPS = fps
        self.human = human

        if board:
            self.set_board(board)

        # close window
        self.running = True

    def loop(self):
        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Gomoku")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
            os.path.join(base_folder, 'back.png')).convert()

        while self.running:
            # timer
            self.clock.tick(self.FPS)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    self.running = False
                # human input
                if self.human and event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    center = (int(mouse_x / self.GRID_WIDTH + 0.5) - 2,
                              int(mouse_y / self.GRID_WIDTH + 0.5) - 2)

                    if center[0] in range(0, self.n) and center[1] in range(0, self.n) \
                            and self.board[center[0]][center[1]] == 0:
                        self.board.execute_move(center, 1)
                        self.human = False

            # draw
            self.draw_background()
            self.draw_chessman()

            # refresh
            pygame.display.flip()

    def draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

        # draw lines
        rect_lines = [
            ((self.GRID_WIDTH, self.GRID_WIDTH),
             (self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.GRID_WIDTH), (self.WIDTH - self.GRID_WIDTH,
                                                  self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH),
             (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.WIDTH - self.GRID_WIDTH, self.GRID_WIDTH),
             (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
        ]
        for line in rect_lines:
            pygame.draw.line(self.screen, self.BLACK, line[0], line[1], 2)

        # draw grid
        for i in range(self.n):
            pygame.draw.line(
                self.screen, self.BLACK,
                (self.GRID_WIDTH * (2 + i), self.GRID_WIDTH),
                (self.GRID_WIDTH * (2 + i), self.HEIGHT - self.GRID_WIDTH))
            pygame.draw.line(
                self.screen, self.BLACK,
                (self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)),
                (self.HEIGHT - self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)))

    def draw_chessman(self):
        # draw chessmen
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] != 0:
                    center = (int(self.GRID_WIDTH * (i + 2)),
                              int(self.GRID_WIDTH * (j + 2)))
                    color = self.WHITE if self.board[i][j] == 1 else self.BALCK
                    pygame.draw.circle(self.screen, color, center,
                                       int(self.GRID_WIDTH / 2.5))

    def set_board(self, board):
        # change the board
        self.n = np.size(board, 0)
        self.board = board
        self.GRID_WIDTH = self.WIDTH / (self.n + 3)

    def close_gui(self):
        # close window
        self.running = False

    def in_turn_human(self):
        # allow human action
        self.human = True
