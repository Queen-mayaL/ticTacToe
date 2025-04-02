from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS
import random
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from React frontend)

# Constants
BOARD_SIZE = 9
PLAYER_X = 1  # AI (X)
PLAYER_O = -1  # Player (O)
EMPTY = 0  # Empty cell

class TicTacToeNN:
    def __init__(self):
        self.input_size = BOARD_SIZE
        self.hidden_size = 18  # Can be adjusted
        self.output_size = BOARD_SIZE
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, state):
        hidden_input = np.dot(state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(final_input)
        return final_output, hidden_output

    def train(self, board, target_q_values, learning_rate=0.1):
        state = np.array(board).reshape(1, BOARD_SIZE)
        final_output, hidden_output = self.forward(state)
        error = target_q_values - final_output
        d_output = error * self.sigmoid_derivative(final_output)
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_output)

        self.weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += state.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    def get_q_values(self, board):
        state = np.array(board).reshape(1, BOARD_SIZE)
        q_values, _ = self.forward(state)
        return q_values.flatten()

    def choose_move(self, board):
        q_values = self.get_q_values(board)
        valid_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]
        valid_q_values = q_values[valid_moves]
        best_move = valid_moves[np.argmax(valid_q_values)]
        return best_move

# Game logic
def check_win(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]               # Diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def is_board_full(board):
    return all(cell != EMPTY for cell in board)

# Initialize the game state
game_board = [EMPTY] * BOARD_SIZE
current_player = PLAYER_X  # AI (X) starts

# Create the TicTacToeNN agent
agent = TicTacToeNN()

@app.route('/game', methods=['GET'])
def get_game_state():
    global game_board, current_player
    
    # If the board is empty, let the AI (X) make the first move
    if game_board == [EMPTY] * BOARD_SIZE:
        move = agent.choose_move(game_board)
        game_board[move] = PLAYER_X
        current_player = PLAYER_O  # Switch turn to Player O
    
    # Convert the board to 'X' and 'O' strings for display
    display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
    
    return jsonify({'board': display_board})


# Global variable to track AI's moves
ai_move_history = []
player_move_history = []

@app.route('/move', methods=['POST'])
def make_move():
    global game_board, current_player, ai_move_history, player_move_history
    data = request.get_json()
    move = data['move']
    
    # Track the move
    if current_player == PLAYER_X:
        ai_move_history.append(move)
    else:
        player_move_history.append(move)
    
    # Validate and apply the move
    if game_board[move] != EMPTY:
        return jsonify({'error': 'Invalid move, spot already taken'}), 400
    
    game_board[move] = current_player
    
    # Check if the player has won
    if check_win(game_board, current_player):
        winner = 'X' if current_player == PLAYER_X else 'O'
        # Train the model after the game ends
        train_after_game(winner)
        display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
        return jsonify({'message': f'Player {winner} wins!', 'board': display_board})
    
    # Check for draw
    if is_board_full(game_board):
        # Train the model after the game ends
        train_after_game('draw')
        display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
        return jsonify({'message': 'It\'s a draw!', 'board': display_board})
    
    # Switch players
    current_player = -current_player  # Switch from X to O, or O to X
    
    # If it's AI's turn, let the agent make a move
    if current_player == PLAYER_X:
        move = agent.choose_move(game_board)
        ai_move_history.append(move)  # Track AI's move
        game_board[move] = PLAYER_X
        
        if check_win(game_board, PLAYER_X):
            # Train after AI wins
            train_after_game('AI')
            display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
            return jsonify({'message': 'AI wins!', 'board': display_board})
        if is_board_full(game_board):
            # Train after draw
            train_after_game('draw')
            display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
            return jsonify({'message': 'It\'s a draw!', 'board': display_board})

        current_player = PLAYER_O  # Switch back to the player

    # Convert game_board from numbers (1, -1, 0) to ['X', 'O', ''] format
    display_board = ['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else '' for cell in game_board]
    
    return jsonify({'message': 'Move made', 'board': display_board})

def train_after_game(result):
    """
    Train the model after each game (win, loss, or draw)
    """
    global ai_move_history, player_move_history, agent
    
    # Determine the reward based on the game outcome
    if result == 'AI':  # AI wins
        reward = 1
    elif result == 'draw':  # Draw
        reward = 0
    else:  # AI loses
        reward = -1
    
    # Use the AI's move history and player move history to train the model
    for move in ai_move_history:
        target_q_values = np.zeros(BOARD_SIZE)
        target_q_values[move] = reward
        agent.train(game_board, target_q_values)
    
    # Clear move history for the next game
    ai_move_history = []
    player_move_history = []

@app.route('/reset', methods=['POST'])
def reset_game():
    global game_board, current_player
    # Reset game state
    game_board = [EMPTY] * BOARD_SIZE
    current_player = PLAYER_X  # AI (X) starts the game
    return jsonify({'message': 'Game reset', 'board': [''] * BOARD_SIZE})





###### IF YOU WANT TO PRE-TRAIN THE MODEL:

# def print_board(board):
#     symbols = {1: 'X', -1: 'O', 0: '.'}
#     for i in range(0, BOARD_SIZE, 3):
#         print(' '.join([symbols[board[i]], symbols[board[i+1]], symbols[board[i+2]]]))

# def is_board_full(board):
#     return all(cell != EMPTY for cell in board)

# def play_game(agent):
#     board = [EMPTY] * BOARD_SIZE
#     current_player = PLAYER_X
    
#     while True:
#         print_board(board)
        
#         if current_player == PLAYER_X:
#             move = agent.choose_move(board)
#         else:
#             move = random.choice([i for i in range(BOARD_SIZE) if board[i] == EMPTY])  # Random move for opponent
        
#         board[move] = current_player
        
#         # Check if the current player has won
#         if check_win(board, current_player):
#             print_board(board)
#             print(f"Player {'X' if current_player == PLAYER_X else 'O'} wins!")
#             return current_player
        
#         if is_board_full(board):
#             print_board(board)
#             print("It's a draw!")
#             return 0
        
#         current_player = -current_player  # Switch player

# def train_agent(agent, episodes=1000):
#     for episode in range(episodes):
#         print(f"Episode {episode + 1}/{episodes}")
#         winner = play_game(agent)
        
#         # Reward system: winning or drawing provides some reward
#         reward = 1 if winner == PLAYER_X else -1 if winner == PLAYER_O else 0
        
#         # Update the agent's Q-values (target Q-values should be updated based on the outcome)
#         agent.train(agent.get_q_values([EMPTY] * BOARD_SIZE), reward)
        
#         # Optionally, print training progress after each episode



if __name__ == '__main__':
    # agent = TicTacToeNN()
    # train_agent(agent, episodes=1000)
    # print("Training complete! Now, you can play against the AI.")

    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT isn't set
    app.run(host='0.0.0.0', port=port)
