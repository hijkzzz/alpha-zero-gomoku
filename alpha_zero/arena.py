import numpy as np
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, board_gui=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            board_gui: gui object

        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.board_gui = board_gui

    def play_game(self):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.get_init_board()

        if self.board_gui:
            self.board_gui.set_board(board)

        it = 0
        last_action = -1
        while self.game.get_game_ended(board) == 2:
            it += 1
            ps, _ = players[cur_player + 1].get_action_prob(board, last_action, cur_player)
            action = np.argmax(ps)
            last_action = action
            board, cur_player = self.game.get_next_state(board, cur_player, action)

            if self.board_gui:
                self.board_gui.set_board(board)

        return self.game.get_game_ended(board)

    def play_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.play_game()
            if gameResult == 1:
                oneWon+=1
            elif gameResult == -1:
                twoWon+=1
            else:
                draws+=1

        # change first player
        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = self.play_game()
            if gameResult == -1:
                oneWon+=1                
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws