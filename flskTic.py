from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS
import random
import os
import joblib

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from React frontend)

# Constants
BOARD_SIZE = 9
PLAYER_X = 1  # Our agent (X)
PLAYER_O = -1  # Opponent (O)
EMPTY = 0  # Empty cell

# Neural Network Architecture
class TicTacToeNN:
    def __init__(self, epsilon=0.1):
        self.input_size = BOARD_SIZE
        self.hidden_size = 36  # Try experimenting with this value
        self.output_size = BOARD_SIZE
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)
        self.epsilon = 0.5  # Increase exploration
    
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

    def get_q_values(self, board):
        state = np.array(board).reshape(1, BOARD_SIZE)
        q_values, _ = self.forward(state)
        return q_values.flatten()

    def choose_move(self, board):
        q_values = self.get_q_values(board)
        print("Q-values for current board:", q_values)  # Debug log

        # Epsilon-greedy exploration: if random value < epsilon, choose a random move
        if random.random() < self.epsilon:
            valid_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]
            best_move = random.choice(valid_moves)
            print(f"Exploring: Chosen random move {best_move}")
        else:  # Exploit: choose the best Q-value move
            valid_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]
            valid_q_values = q_values[valid_moves]
            print("Valid moves and corresponding Q-values:", list(zip(valid_moves, valid_q_values)))  # Debug log
            best_move = valid_moves[np.argmax(valid_q_values)]
            print(f"Exploiting: Chosen best move {best_move}")

        return best_move

    
    def train(self, board, target_q_values, learning_rate=0.1):
        # Convert board to array (flattened)
        state = np.array(board).reshape(1, BOARD_SIZE)

        # Forward pass
        final_output, hidden_output = self.forward(state)

        # Print Q-values before training
        print("Before training Q-values:", final_output.flatten())

        # Compute the error (target Q-value - predicted Q-value)
        error = target_q_values - final_output

        # Backpropagation
        d_output = error * self.sigmoid_derivative(final_output)
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_output)

        # Update weights and biases
        self.weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += state.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        # Print Q-values after training
        final_output, _ = self.forward(state)
        print("After training Q-values:", final_output.flatten())

    
    def train_episode(self, episode_moves, rewards):
        """
        Accumulate all moves and their rewards, then train after the episode.
        episode_moves: List of moves made during the episode
        rewards: Corresponding rewards for each move in the episode
        """
        # Iterate over all moves in the episode
        for move, reward in zip(episode_moves, rewards):
            board, target_q_values = move
            self.train(board, target_q_values, reward)

# Check if a player has won
def check_win(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Check if the board is full
def is_board_full(board):
    return all(cell != EMPTY for cell in board)


def load_model_with_joblib(filename="tictactoe_model.pkl"):
    agent = joblib.load(filename)
    print("Loaded model weights_input_hidden:", agent.weights_input_hidden)
    print("Loaded model weights_hidden_output:", agent.weights_hidden_output)
    return agent


def load_model():
    global agent
    try:
        agent = load_model_with_joblib("tictactoe_model.pkl")  # Load once when the server starts
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        agent = TicTacToeNN()  # Fall back to a new agent if loading fails


# @app.route('/get_move', methods=['POST'])
# def get_move():
#     data = request.get_json()
#     board = data.get('board', [])
    
#     # Check for invalid board values
#     if len(board) != 9 or any(val not in [0, 1, -1] for val in board):
#         return jsonify({"error": "Invalid board format. Board must contain only 0, 1, or -1."}), 400
#     agent = load_model_with_joblib("tictactoe_model.pkl")  # Load the model here

#     # agent = TicTacToeNN()  # Instantiate the agent
#     move = agent.choose_move(board)  # Get the move from the agent
    
#     return jsonify({"move": move})  # Return the move in a JSON response

@app.route('/get_move', methods=['POST'])
def get_move():
    data = request.get_json()
    board = data.get('board', [])
    
    # Check for invalid board values
    if len(board) != 9 or any(val not in [0, 1, -1] for val in board):
        return jsonify({"error": "Invalid board format. Board must contain only 0, 1, or -1."}), 400

    # Get the current model (do not reload here, use the one loaded at server startup)
    global agent
    move = agent.choose_move(board)  # Get the move from the agent
    
    return jsonify({"move": move})  # Return the move in a JSON response


@app.route('/check_status', methods=['POST'])
def check_status():
    data = request.get_json()
    board = data['board']
    current_player = data['current_player']
    
    if check_win(board, 1):  # Check if X wins
        return jsonify({"status": "win", "player": 1})  # X wins
    
    elif check_win(board, -1):  # Check if O wins
        return jsonify({"status": "win", "player": -1})  # O wins
    
    elif is_board_full(board):  # Check if it's a draw (board full and no winner)
        return jsonify({"status": "draw"})
    
    return jsonify({"status": "continue"})  # The game continues






###### IF YOU WANT TO PRE-TRAIN THE MODEL:

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for i in range(0, BOARD_SIZE, 3):
        print(' '.join([symbols[board[i]], symbols[board[i+1]], symbols[board[i+2]]]))

def is_board_full(board):
    return all(cell != EMPTY for cell in board)

def play_game():
    agent = load_model_with_joblib("tictactoe_model.pkl")  # Load the model here
    board = [EMPTY] * BOARD_SIZE
    current_player = PLAYER_X
    
    while True:
        print_board(board)
        
        if current_player == PLAYER_X:
            move = agent.choose_move(board)
        else:
            move = random.choice([i for i in range(BOARD_SIZE) if board[i] == EMPTY])  # Random move for opponent
        
        board[move] = current_player
        
        # Check if the current player has won
        if check_win(board, current_player):
            print_board(board)
            print(f"Player {'X' if current_player == PLAYER_X else 'O'} wins!")
            return current_player
        
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            return 0
        
        current_player = -current_player  # Switch player

# def train_agent(agent, episodes=1000):
#     for episode in range(episodes):
#         print(f"Episode {episode + 1}/{episodes}")
#         board = [EMPTY] * BOARD_SIZE
#         current_player = PLAYER_X
#         episode_moves = []
#         rewards = []

#         while True:
#             if current_player == PLAYER_X:
#                 move = agent.choose_move(board)
#             else:
#                 move = random.choice([i for i in range(BOARD_SIZE) if board[i] == EMPTY])  # Random move for opponent
            
#             board[move] = current_player
#             episode_moves.append((board.copy(), agent.get_q_values(board)))  # Save the board and the Q-values for each move

#             # Check if the current player has won
#             if check_win(board, current_player):
#                 reward = 1 if current_player == PLAYER_X else -1
#                 rewards.append(reward)
#                 print(f"Player {'X' if current_player == PLAYER_X else 'O'} wins!")
#                 break

#             # Check for a draw
#             if is_board_full(board):
#                 rewards.append(0)  # No reward, it's a draw
#                 print("It's a draw!")
#                 break

#             # Switch player
#             current_player = -current_player  # Alternate players

#         # Train the agent after the episode ends
#         agent.train_episode(episode_moves, rewards)

def train_agent(agent, episodes=1000):
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        board = [EMPTY] * BOARD_SIZE
        current_player = PLAYER_X
        
        while True:
            if current_player == PLAYER_X:
                move = agent.choose_move(board)
            else:
                move = random.choice([i for i in range(BOARD_SIZE) if board[i] == EMPTY])  # Random move for opponent
            
            board[move] = current_player

            # Check if the current player has won
            if check_win(board, current_player):
                reward = 1 if current_player == PLAYER_X else -1
                agent.train(board, reward)  # Update Q-values for this game
                print(f"Player {'X' if current_player == PLAYER_X else 'O'} wins!")
                break

            # Check for a draw
            if is_board_full(board):
                agent.train(board, 0)  # No reward, it's a draw
                print("It's a draw!")
                break

            # Switch player
            current_player = -current_player  # Alternate players


def save_model_with_joblib(agent, filename="tictactoe_model.pkl"):
    joblib.dump(agent, filename)
    print("Model saved to", filename)


def train_and_save_model():
    agent = TicTacToeNN()
    train_agent(agent, episodes=20000)  # Train the agent for 1000 episodes
    save_model_with_joblib(agent)


def play_game_Manual():
    agent = load_model_with_joblib("tictactoe_model.pkl")  # Load the model here

    board = [EMPTY] * BOARD_SIZE
    current_player = PLAYER_X  # The AI always starts as 'X'
    
    while True:
        print_board(board)
        
        if current_player == PLAYER_X:
            # AI move (automatically selects based on Q-values)
            move = agent.choose_move(board)
            print(f"AI (X) selects position {move}...")
        else:
            # Player move (manual input for 'O')
            valid_move = False
            while not valid_move:
                try:
                    move = int(input("Enter your move (0-8): "))
                    if board[move] == EMPTY:
                        valid_move = True
                    else:
                        print("Invalid move! The spot is already taken.")
                except ValueError:
                    print("Invalid input! Please enter an integer between 0 and 8.")
                except IndexError:
                    print("Invalid move! Please enter a number between 0 and 8.")
        
        # Make the move
        board[move] = current_player
        
        # Check if the current player has won
        if check_win(board, current_player):
            print_board(board)
            print(f"Player {'X' if current_player == PLAYER_X else 'O'} wins!")
            return current_player
        
        # Check if the board is full (draw condition)
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            return 0
        
        # Switch players
        current_player = -current_player  # Alternate between X and O


if __name__ == '__main__':
    # agent = TicTacToeNN()
    # train_agent(agent, episodes=1000)
    # print("Training complete! Now, you can play against the AI.")
    # train_and_save_model()
    # play_game()
    # play_game_Manual()
    load_model()
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT isn't set
    app.run(host='0.0.0.0', port=port)