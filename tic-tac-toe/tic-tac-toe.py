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
        
        self.history = dict.fromkeys(['Moves', 'Board'])

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
        self.history = dict.fromkeys(['Moves', 'Board'])
        
    def display(self):
        # Display the current state of the board
        print(self.board)
    
    def get_possible_actions(self, state, moves, move):
        # Get available actions (empty cells) in the given state
        actions = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    new_env_board = state.copy()
                    new_env_moves = moves.copy()
                    # i = j = 0
                    new_env_moves[i][j] = move
                    
                    new_board_tuple = tuple(map(tuple, new_env_board))
                    new_moves_tuple = tuple(map(tuple, new_env_moves))
                    
                    if (self.history['Board'] is None or new_board_tuple not in self.history['Board']) or \
                    (self.history['Moves'] is None or new_moves_tuple not in self.history['Moves']):
                        actions.append((i, j))

        return actions
    
    def is_empty(self, row, col):
        # Check if a cell is empty
        return self.board[row, col] == ' ' and self.moves[row, col] == ' '

    def is_winner(self, symbol):
        # Check if the current player with the given symbol has won
        for i in range(3):
            # Check rows and columns
            if all(self.board[i, j] == symbol for j in range(3)) or all(self.board[j, i] == symbol for j in range(3)):
                return True

        # Check diagonals
        if all(self.board[i, i] == symbol for i in range(3)) or all(self.board[i, 2 - i] == symbol for i in range(3)):
            return True

        return False

    def is_draw(self):
        # Check if the game is a draw
        return ' ' not in self.board.flatten() and not any(self.is_winner(symbol) for symbol in ['X', 'O'])

    def make_move(self, row, col, symbol, move):
        # Make a move on the board
        self.board[row, col] = symbol
        self.moves[row, col] = move
    
    def is_winner_move(self, row, col, state):
        # Check if the current move is a winning move
        temp_board = np.copy(self.board)
        temp_moves = np.copy(self.moves)
        temp_board[row, col] = state[row][col]
        temp_moves[row, col] = 'X' if state[row][col] == ' ' else 'O'
        return self.is_winner(temp_moves[row, col])
        
# Define the Q-learning agent
class QLearningAgent:
    def __init__(self):
        self.q_table = {}  # You can use a dictionary for the Q-table
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
        print(agent.q_table.get((state_tuple, action_tuple), 0.0))
        return self.q_table.get((state_tuple, action_tuple), 0.0)

    def choose_action(self, state, available_actions, moves):
        winning_moves = [action for action in available_actions if env.is_winner_move(action[0], action[1], state)]
        other_moves = [action for action in available_actions if action not in winning_moves]

        # if np.random.rand() < self.epsilon:
        if np.random.rand() < agent.epsilon:
            if winning_moves:
                action = random.choice(winning_moves)
            else:
                action = random.choice(other_moves)
        else:
            state_tuple = tuple(map(tuple, state))
            #q_values = [self.get_q_value(state, action) for action in available_actions]
            # q_values = [agent.get_q_value(state, action) for action in available_actions] #action = (0,0)
            q_values = [agent.q_table.get((state_tuple, action), 0.0) for action in available_actions] #action = (0,0)
            action_index = np.argmax(q_values)
            action = available_actions[action_index]

        self.actions_taken.add(action)
        self.moves_taken.add(moves[action[0], action[1]])

        if env.is_winner_move(action[0], action[1], state):
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
        self.q_table[(state, action)] = new_q

        # Reset actions_taken for the next episode
        self.actions_taken = set()

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
        # print("Episode: " + str(episode + 1))
        env.reset()
        # if episode % evaluation_interval == 0 and episode != 0:
        #     print(f"Evaluating model after {episode} episodes")
        #     evaluate_agent(agent, env)
        
        # state = tuple(map(tuple, env.board))
        state = env.board.copy()
        total_reward = 0
        symbol = 'X'
        moves = np.zeros((3, 3))  # Initialize moves matrix for the current move
        move = 1
        
        while True:

            reward = 0
            print("State:", state)
            
            available_actions = env.get_possible_actions(state, moves, move)
            print("Available actions:", available_actions)

            if not available_actions:
                break

            action = agent.choose_action(state, available_actions, moves)
            print("Chosen action:", action)

            row, col = action
            env.make_move(row, col, symbol, move)
            # env.make_move(1, 1, 'X', 1)

            # next_state = tuple(map(tuple, env.board.copy()))
            next_state = env.board.copy()
            print("Next state:", next_state)

            reward = calculate_reward(env)
            print("Reward:", reward)

            agent.update_q_value(state, action, reward, next_state, moves, move)

            total_reward += reward

            if env.is_winner('X') or env.is_winner('O') or env.is_draw():
                env.history['Moves'].append(tuple(map(tuple, moves)))
                env.history['Board'].append(tuple(map(tuple, state)))
                break

            symbol = 'O' if symbol == 'X' else 'X'
            state = next_state
            moves[row, col] = move  # Update moves matrix
            move += 1
            print("Moves:", tuple(map(tuple, moves)))

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
        moves = np.zeros((3, 3))  # Initialize moves matrix for the current move    
        
        while True:

            available_actions = env.get_possible_actions(state, moves, move)
            print("Available Actions:", available_actions)
            
            action = agent.choose_action(state, available_actions, moves)  # Pass a dummy matrix for moves
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
