# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:46:36 2024

@author: ilepoutre
"""

# https://github.com/dennybritz/reinforcement-learning
# https://github.com/AI4Finance-Foundation/FinRL

# https://towardsdatascience.com/reinforcement-learning-explained-visually-part-4-q-learning-step-by-step-b65efb731d3e

# https://medium.com/@sadeepari/beyond-the-sandbox-building-your-own-reinforcement-learning-grid-world-65dbd2a1705d



import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, shape=(5, 5)):
        self.shape = shape
        self.grid = np.zeros(shape)
        self.agent_position = (0, 0)
        self.agent_state = 0
        self.goal_position = (shape[0] - 1, shape[1] - 1)
        self.obstacle_positions = [(1, 1), (2, 2), (3, 3)]  # Example obstacle positions

    def get_state(self, position):
        i, j = position
        return i * self.shape[1] + j

    def reset(self):
        self.agent_position = (0, 0)
        self.state = self.get_state((0, 0))
        
        return self.agent_position, self.state

    def step(self, action):
        reward = -1  # Default reward for each step
        done = False
        
        new_position = self._get_new_position(action)
        
        # Check if the new position is valid
        if self._is_valid_position(new_position):
            self.agent_position = new_position
            self.agent_state = self.get_state(self.agent_position)
        else:
            reward -= 5  # Additional penalty for hitting a wall
        
        # Check if the agent reached the goal
        if self.agent_position == self.goal_position:
            reward += 10
            done = True
        
        # Check if the agent hit an obstacle
        if self.agent_position in self.obstacle_positions:
            reward -= 10
            done = True
        
        return self.agent_state, reward, done
        # return self.agent_position, reward, done

    def _get_new_position(self, action):
        x, y = self.agent_position
        if action == 0:  # Up
            return (max(0, x - 1), y)
        elif action == 1:  # Down
            return (min(self.shape[0] - 1, x + 1), y)
        elif action == 2:  # Left
            return (x, max(0, y - 1))
        elif action == 3:  # Right
            return (x, min(self.shape[1] - 1, y + 1))

    def _is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.shape[0] and 0 <= y < self.shape[1] and position not in self.obstacle_positions

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.grid, cmap='gray', origin='upper')

        # Plot obstacles
        for obstacle_pos in self.obstacle_positions:
            plt.scatter(obstacle_pos[1], obstacle_pos[0], color='red', marker='s', s=100)

        # Plot agent
        plt.scatter(self.agent_position[1], self.agent_position[0], color='blue', marker='o', s=100)

        # Plot goal
        plt.scatter(self.goal_position[1], self.goal_position[0], color='green', marker='o', s=100)

        plt.title('Grid World')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    # https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4-triangles
    
    def plot_with_Q(self, agent_q_table=None):
        plt.figure(figsize=(8, 6))
    
        # Plot grid
        plt.imshow(self.grid,cmap='gray', origin='upper') # , cmap='gray', origin='upper')
        
        # Plot obstacles
        for obstacle_pos in self.obstacle_positions:
            plt.scatter(obstacle_pos[1], obstacle_pos[0], color='red', marker='s', s=100)
    
        # Plot goal
        plt.scatter(self.goal_position[1], self.goal_position[0], color='green', marker='o', s=100)
    
        # Plot Q-values for each state
        if agent_q_table is not None:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    state = (i, j)
                    if state in self.obstacle_positions or state == self.goal_position:
                        continue
                    q_values = agent_q_table[i * self.shape[1] + j]
                    max_q_value = np.max(q_values)
                    for action, q_value in enumerate(q_values):
                        color = 'gray'
                        if q_value == max_q_value:
                            color = 'blue' if action == 0 else 'orange' if action == 1 else 'green' if action == 2 else 'red'
                        plt.fill_between([j, j + 1], self.shape[0] - i, self.shape[0] - i - 1, color=color, alpha=q_value / max_q_value * 0.8)
                        
        plt.title('Grid World with Q-values')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()


    # https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4-triangles
    def plot_with_Q_2():
        from matplotlib import pyplot as plt
        from matplotlib.tri import Triangulation
        import numpy as np

        def create_demo_data(M, N):
            # create some demo data for North, East, South, West
            # note that each of the 4 arrays can be either 2D (N by M) or 1D (N*M)
            # M columns and N rows
            valuesN = np.repeat(np.abs(np.sin(np.arange(N))), M)
            valuesE = np.arange(M * N) / (N * M)
            valuesS = np.random.uniform(0, 1, (N, M))
            valuesW = np.random.uniform(0, 1, (N, M))
            return [valuesN, valuesE, valuesS, valuesW]
        
        def triangulation_for_triheatmap(M, N):
            xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
            xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
            x = np.concatenate([xv.ravel(), xc.ravel()])
            y = np.concatenate([yv.ravel(), yc.ravel()])
            cstart = (M + 1) * (N + 1)  # indices of the centers
        
            trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                          for j in range(N) for i in range(M)]
            trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                          for j in range(N) for i in range(M)]
            trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                          for j in range(N) for i in range(M)]
            trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                          for j in range(N) for i in range(M)]
            return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]
        
        M, N = 5, 4  # e.g. 5 columns, 4 rows
        values = create_demo_data(M, N)
        triangul = triangulation_for_triheatmap(M, N)
        cmaps = ['Blues', 'Greens', 'Purples', 'Reds']  # ['winter', 'spring', 'summer', 'autumn']
        norms = [plt.Normalize(-0.5, 1) for _ in range(4)]
        fig, ax = plt.subplots()
        imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
                for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]
        
        ax.set_xticks(range(M))
        ax.set_yticks(range(N))
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.show()




# Q-learning agent
class QLearningAgent:
    def __init__(self, n_actions, n_states, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        
        # state, action, reward, next_state = 
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


# Training the agent
def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        position, state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}: Goal reached")
            print(agent.q_table)

# Main function
if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent(n_actions=4, n_states=np.prod(env.shape))
    # train_agent(env, agent, 100)




