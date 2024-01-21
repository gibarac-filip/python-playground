# Import necessary libraries
import numpy as np
import random
from tqdm import tqdm

# Define the Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = np.array([[' ', ' ', ' '],
                               [' ', ' ', ' '],
                               [' ', ' ', ' ']])
        
        self.moves = np.array([[' ', ' ', ' '],
                               [' ', ' ', ' '],
                               [' ', ' ', ' ']])
        
        self.history = {'Moves': [], 'Board': []}

    def reset(self):
        # Reset the board to an initial state
        self.board = np.array([[' ', ' ', ' '],
                               [' ', ' ', ' '],
                               [' ', ' ', ' ']])
        
        # Reset the moves to an initial state
        self.moves = np.array([[' ', ' ', ' '],
                               [' ', ' ', ' '],
                               [' ', ' ', ' ']])
                
    def reset_history(self):
        self.history = {'Moves': [], 'Board': []}
        
    def display(self):
        # Display the current state of the board
        print(self.board)
    
    def is_empty(self, row, col, state=None,  moves=None):
        # If a state is provided, use it; otherwise, use self.board
        board_to_check = state if state is not None else self.board
        moves_to_check = moves if moves is not None else self.moves
        
        # Check if a cell is empty
        return board_to_check[row, col] == ' ' and moves_to_check[row, col] == ' '

    def is_winner(self, symbol, state=None):
        # If a state is provided, use it; otherwise, use self.board
        board_to_check = state if state is not None else self.board

        # Check if the current player with the given symbol has won
        for i in range(3):
            # Check rows and columns
            if all(board_to_check[i, j] == symbol for j in range(3)) or all(board_to_check[j, i] == symbol for j in range(3)):
                return True

        # Check diagonals
        if all(board_to_check[i, i] == symbol for i in range(3)) or all(board_to_check[i, 2 - i] == symbol for i in range(3)):
            return True

        return False

    def is_draw(self):
        # Check if the game is a draw
        return ' ' not in self.board.flatten() and not any(self.is_winner(symbol) for symbol in ['X', 'O'])

    def make_move(self, row, col, symbol, move):
        # Make a move on the board
        self.board[row, col] = symbol
        self.moves[row, col] = move
    
    def is_winning_move(self, row, col, state, symbol, moves, move):
        temp_board = np.copy(state)
        temp_moves = np.copy(moves)
        temp_board[row, col] = symbol
        temp_moves[row, col] = move
        
        return self.is_winner(symbol, temp_board)
        
    def get_possible_actions(self, state, moves, move):
        # Get available actions (empty cells) in the given state
        actions = []
        for i in range(3):
            for j in range(3):
                if state[i, j] == ' ':
                    new_env_board = state.copy()
                    new_env_moves = moves.copy()
                    new_env_moves[i][j] = move

                    if (self.history['Board'] is None or not any(np.array_equal(new_env_board, board) for board in self.history['Board'])) or \
                    (self.history['Moves'] is None or not any(np.array_equal(new_env_moves, moves) for moves in self.history['Moves'])):
                        actions.append((i, j))

        return actions

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self):
        self.q_table = {} 
        self.learning_rate = 0.15
        self.discount_factor = 0.85
        self.epsilon = 0.01  # Exploration-exploitation trade-off
        self.actions_taken = set()  # Track actions taken in the current episode
        self.moves_taken = set()  # Track moves taken in the current episode
        self.winning_moves_taken = set()  # Track winning moves taken in the current episode
    
    def random_initialize_q_table(self, all_possible_states, all_possible_actions):
        # Randomly initialize the Q-table
        for state in all_possible_states:
            for action in all_possible_actions:
                self.q_table[(state, action)] = random.uniform(0, 1)  # Assign a random value between 0 and 1

    def get_q_value(self, state, action):
        # Convert state and action to tuples
        state_tuple = tuple(map(tuple, state))
        action_tuple = tuple(action)
        
        # Get the Q-value for a state-action pair from the Q-table
        # Initialize with a default value if not present
        return self.q_table.get((state_tuple, action_tuple), 0.0)

    def choose_action(self, state, available_actions, moves, symbol, move):
        winning_moves = [action for action in available_actions if env.is_winning_move(action[0], action[1], state, symbol, moves, move)]
        other_moves = [action for action in available_actions if action not in winning_moves]

        # if np.random.rand() < self.epsilon:
        if np.random.rand() < agent.epsilon:
            if winning_moves:
                action = random.choice(winning_moves)
            else:
                action = random.choice(other_moves)
        else:
            state_tuple = tuple(map(tuple, state))
            q_values = [agent.q_table.get((state_tuple, action), 0.0) for action in available_actions]
            action_index = np.argmax(q_values)
            action = available_actions[action_index]

        self.actions_taken.add(action)
        self.moves_taken.add(moves[action[0], action[1]])

        if env.is_winning_move(action[0], action[1], state, symbol, moves, move):
            self.winning_moves_taken.add(action)

        return action

    def update_q_value(self, state, action, reward, next_state, moves, move):
        # Update the Q-value using the Q-learning update rule
        current_q = self.get_q_value(state, action)

        # Check if there are possible actions in the next state
        possible_actions_next = env.get_possible_actions(next_state, moves, move)

        if possible_actions_next:
            max_future_q = max([self.get_q_value(next_state, next_action) for next_action in possible_actions_next])
        else:
            max_future_q = 0  # If no possible actions, set max_future_q to 0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[(tuple(map(tuple, state)), action)] = new_q
        return self.q_table.get((tuple(map(tuple, state)), action), 0.0)


# Function to calculate reward for result of game
def calculate_reward(env):
    if env.is_winner('X'):
        return 1
    elif env.is_winner('O'):
        return -1
    elif env.is_draw():
        return -0.1
    else:
        return 0

# Main function
def main():
    # Create instances of the environment and agent
    env = TicTacToe()
    agent = QLearningAgent()

    # Training loop
    global episodes
    episodes = 2  # Max potential actions allowed for  3x3 game, 362880
    # evaluation_interval = 100000  # Set your desired evaluation interval
    
    # Initialize the Q-table with random values
    # all_possible_states = [tuple(map(tuple, np.zeros((3, 3))))]  # Add all possible states here
    # all_possible_actions = [(i, j) for i in range(3) for j in range(3)]  # Add all possible actions here
    # agent.random_initialize_q_table(all_possible_states, all_possible_actions)
    print("Starting training")
    
    total_rewards = list()
    
    for episode in tqdm(range(episodes), desc="Training", unit=" episodes"):
        env.reset()
        # if episode % evaluation_interval == 0 and episode != 0:
        #     print(f"Evaluating model after {episode} episodes")
        #     evaluate_agent(agent, env)
        
        state = env.board.copy()
        total_reward = 0
        symbol = 'X'
        moves = np.zeros((3, 3))
        move = 1
        
        while True:

            reward = 0
            print("State:", state)
            
            available_actions = env.get_possible_actions(state, moves, move)
            print("Available actions:", available_actions)

            if not available_actions:
                break

            action = agent.choose_action(state, available_actions, moves, symbol, move)
            print("Chosen action:", action)

            row, col = action
            env.make_move(row, col, symbol, move)

            next_state = env.board.copy()
            print("Next state:", next_state)

            reward = calculate_reward(env)
            print("Reward:", reward)

            agent.update_q_value(state, action, reward, next_state, env.moves, move)

            total_reward += reward

            if env.is_winner('X') or env.is_winner('O') or env.is_draw():
                env.history['Moves'].append(tuple(map(tuple, moves)))
                env.history['Board'].append(tuple(map(tuple, state)))
                break

            symbol = 'O' if symbol == 'X' else 'X'
            state = next_state
            moves[row, col] = move
            move += 1

        total_rewards.append(total_reward)
        
        # Print progress occasionally
        # if episode % 10000 == 0:
        #    print(f"Episode: {episode}, Total Reward: {total_reward}")
    print("Total Reward:", total_rewards)
    print("Training complete.")
    
    # Print or plot the total rewards over episodes
    # import matplotlib.pyplot as plt

    # Scale the total rewards by a factor of 10,000 for better readability
    # scaled_rewards = [reward / 10000 for reward in total_rewards]

    # plt.plot(range(episodes), scaled_rewards)
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Reward (in 10,000s)')
    # plt.title('Training Progress')
    # plt.show()

def evaluate_agent(agent, env, player_symbol='X'):
    # Evaluate the agent without updating the Q-table
    print("Evaluating model")
    
    total_wins = 0
    total_rewards = 0
    
    for _ in tqdm(range(episodes), desc="Evaluating", unit=" episodes"):
        env.reset()
        state = tuple(map(tuple, env.board))
        print("State:", state)
        move = 1
        moves = np.zeros((3, 3))  
        
        while True:

            available_actions = env.get_possible_actions(state, moves, move)
            print("Available Actions:", available_actions)
            
            action = agent.choose_action(state, available_actions, moves, player_symbol, move)
            print("Action Chosen:", action)
            
            # Take the chosen action with the specified player symbol
            row, col = action
            env.make_move(row, col, player_symbol, move)
        
            next_state = tuple(map(tuple, env.board.copy()))
            print("Next state:", next_state)

            if env.is_winner('X') or env.is_winner('O') or env.is_draw():
                break
            
            player_symbol = 'O' if player_symbol == 'X' else 'X'
            state = next_state
            moves[row, col] = move  # Update moves matrix
            move += 1

        total_rewards += calculate_reward(env)
        print("Total Rewards:", total_rewards)

        if env.is_winner(player_symbol):
            total_wins += 1

    winning_rate = (total_wins / episodes) * 100
    average_reward = total_rewards / episodes

    print(f"Winning Rate: {winning_rate:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")

# Call the main function
if __name__ == "__main__":
    env = TicTacToe()
    agent = QLearningAgent()
    all_possible_states = [tuple(map(tuple, np.zeros((3, 3))))]  # Add all possible states here
    all_possible_actions = [(i, j) for i in range(3) for j in range(3)]  # Add all possible actions here
    agent.random_initialize_q_table(all_possible_states, all_possible_actions)

    # Training phase
    main()

    # Evaluation phase
    # evaluate_agent(agent, env)
