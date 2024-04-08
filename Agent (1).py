#!/usr/bin/env python
# coding: utf-8

# In[1]:


from aima3.agents import *
from aima3.notebook import psource


# # Task#01: Simple Reflex Agents
# Consider an interactive and cognitive environment (ICE) in which a smart camera is
# monitoring robot movement from one location to another. Let a robot be at location A for
# some time instant and then moves to point B and eventually reaches at point C and so on
# and so forth shown in the Fig. Develop a Python code to calculate a distance between
# reference point R (4, 0) of a camera and A, B, and C and N number of locations.

# In[13]:


import math

def calculate_distance(reference_point, locations):
    distances = {}
    for location_name, x_y_point in locations.items():
        distance = math.sqrt((R[0] - x_y_point[0])**2 + (R[1] - x_y_point[1])**2)
        distances[location_name] = distance
    return distances

R=(4, 0)
locations = {
    'A': (1, 1),
    'B': (2, 3),
    'C': (5, 5)
}

distances = calculate_distance(reference_point,locations)

for location, distance in distances.items():
    print("distance between Referenc point and {} : {}".format(location,distance))


# # Task#02 Simple Reflex Agents
# Consider a scenario, cameras placed on every side of the car — front, rear, left and right —
# to stitch together a 360-degree view of the environment. For a three-lane road a car is
# moving on a middle lane, consider the below scenario
# • If the front camera detects the object within range of 8 meters breaks are applied
# automatically.
# • If the left camera detects the object within range of 2 meters car moves to the right lane.
# • If the right camera detects the object within range of 2 meters car moves to the left lane.
# • For parking the car if the rear camera detects the object within 5 cm breaks are applied.

# In[18]:


class Car_Cameras:
    def __init__(self):
        self.front_camera_range = 8
        self.left_camera_range = 2
        self.right_camera_range = 2
        self.rear_camera_range = 0.05 

    def front_camera(self,distance):
        if distance <= self.front_camera_range:
            return True
        return False

    def left_camera(self,distance):
        if distance <= self.left_camera_range:
            return True
        return False

    def right_camera(self,distance):
        if distance <= self.right_camera_range:
            return True
        return False

    def rear_camera(self,distance):
        if distance <= self.rear_camera_range:
            return True
        return False
    
agent = Car_Cameras()

front_distance = 5
left_distance = 1
right_distance = 3
rear_distance = 0.02  

if agent.front_camera(front_distance):
    print("front camera detects an object, applying brakes")
if agent.left_camera(left_distance):
    print("left camera detects an object, moving to the right lane")
if agent.right_camera(right_distance):
    print("right camera detects an object, moving to the left lane")
if agent.rear_camera(rear_distance):
    print("rear camera detects an object while parking, applying brakes")


# # Task#03 Simple Reflex Agents
# Consider the following scenario where the UAV receives temperature data from the installed
# sensors in a residential area. Assume that there are nine sensors installed that are measuring
# temperature in centigrade. Develop a Python code to calculate the average temperature
# in F.

# In[43]:


def Farhenhiet(centigrade):
    UAV = {}
    
    for sensors, temp in centigrade.items():
        temp_in_Fahrenheit = ((9/5) * temp) + 32
        UAV[sensors] = temp_in_Fahrenheit
    return UAV
    
def avg_temp(temperatures):
    average_temp = sum(temperatures.values()) / len(temperatures)
    return average_temp
        
centigrade = {
    'A': 34,
    'B': 32,
    'C': 20,
    'D': 22,
    'E': 44,
    'F': 43,
    'G': 29,
    'H': 30,
    'I': 48
}

temperatures = Farhenhiet(centigrade)
average_temp_F = avg_temp(temperatures)


for sensor, temp_in_Fahrenheit in temperatures.items():
    print("sensor {} temperature in fahrenheit {}F".format(sensor,temp_in_Fahrenheit))
    
print("\naverage temperature in farhenhiet {}F".format(average_temp_F))


# Task#04 Model Based Reflex Agents
# An AI startup has approached you to write a program for their automatic vacuum cleaner.
# For the vacuum cleaner the room appears to be a matrix of n * m. Each index referred to as
# a cell of the matrix is valued as dirty “D”, clean “C”. The cells which are occupied by the stuff
# in the room are blocked and valued “B”. The vacuum can move in all four directions (up,
# down, left, right), and if the cell status is D, it will clean the cell and change the status to “C”,
# if the cell status is either C, it will not enter the cell. The vacuum will stop working if the
# surrounding of its positions is cleaned (Up, Left, Right, Down), i.e., the status of all the cells is
# either C. The vacuum may start cleaning the room from the first cell (0, 0) or any random
# location. You will trace the path of the vacuum and display at the end of the program.

# In[1]:


class VacuumEnvironment:
    def __init__(self, grid):
        self.grid = grid
        self.n = len(grid)
        self.m = len(grid[0]) if self.n else 0
        self.path = []

    def is_dirty(self, location):
        return self.grid[location[0]][location[1]] =='D'

    def is_clean(self, location):
        return self.grid[location[0]][location[1]] =='C'

    def is_blocked(self, location):
        return self.grid[location[0]][location[1]] =='B'

    def clean(self, location):
        if self.is_dirty(location):
            self.grid[location[0]][location[1]] ='C'

    def move(self,agent_location,direction):
        x,y = agent_location
        if direction == 'up' and x > 0:
            return (x-1,y)
        elif direction == 'down' and x < self.n - 1:
            return (x+1,y)
        elif direction == 'left' and y > 0:
            return (x,y-1)
        elif direction == 'right' and y < self.m - 1:
            return (x,y+1)
        return agent_location

class ModelBasedVacuumAgent:
    def __init__(self, environment, start_location=(0, 0)):
        self.environment = environment
        self.location = start_location
        self.environment.path.append(self.location)

    def choose_action(self):
        if self.environment.is_dirty(self.location):
            return 'clean'
        for direction in ['up','right','down','left']:
            new_location = self.environment.move(self.location, direction)
            if new_location != self.location and not self.environment.is_clean(new_location):
                return direction
        return 'NoOp'

    def act(self):
        action = self.choose_action()
        if action == 'clean':
            self.environment.clean(self.location)
        else:
            self.location = self.environment.move(self.location, action)
            self.environment.path.append(self.location)
        return action

    def run(self):
        while True:
            action = self.act()
            if action == 'NoOp':
                break

grid = [
    ['D','D','D','D','D'],
    ['C','D','B','C','D'],
    ['D','D','C','C','C'],
    ['D','D','D','D','D'],
    ['D','D','D','D','D']
]

environment = VacuumEnvironment(grid)
agent = ModelBasedVacuumAgent(environment)
agent.run()

print("path of the vacuum cleaner ", environment.path)


#  Task#5 Tic tac toe is a very simple two-player game where both the player gets to choose any of
# the symbols between X and O. This game is played on a 3X3 grid board and one by one
# each player gets a chance to mark its respective symbol on the empty spaces of the grid.
# Once a player is successful in marking a strike of the same symbol either in the horizontal,
# vertical or diagonal way as shown in the picture below is created that player wins the game
# else the game goes on a draw if all the spots are filled. write a program for a tic tac toe game for a single-player game, in this program user plays against a computer agent. computer agent and player get multiple chances to mark their
# respective symbol on the empty spaces of the grid. The goal of the computer agent is to win
# the game. You can assign (X or O) to a computer agent. On every move, the computer must
# strike its marking where necessary to win the game. If there is no winning move on any strike
# then the computer marks anywhere on the board.

# In[9]:


import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_win(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]
    return [player, player, player] in win_conditions

def get_empty_cells(board):
    return [(x, y) for x in range(3) for y in range(3) if board[x][y] == " "]

def make_move(board, cell, player):
    board[cell[0]][cell[1]] = player

def computer_move(board, computer):
    empty_cells = get_empty_cells(board)
    for cell in empty_cells:
        board_copy = [row[:] for row in board]
        make_move(board_copy, cell, computer)
        if check_win(board_copy, computer):
            return cell

    for cell in empty_cells:
        board_copy = [row[:] for row in board]
        make_move(board_copy, cell, 'X' if computer == 'O' else 'O')
        if check_win(board_copy, 'X' if computer == 'O' else 'O'):
            return cell

    return random.choice(empty_cells) if empty_cells else None

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    player = "X"
    computer = "O"
    current_player = "X"

    while True:
        print_board(board)
        if current_player == player:
            x, y = map(int, input("Enter row and column (0, 1, 2): ").split())
            make_move(board, (x, y), player)
        else:
            cell = computer_move(board, computer)
            if cell:
                make_move(board, cell, computer)
            else:
                print("Draw!")
                break

        if check_win(board, current_player):
            print_board(board)
            print(f"{current_player} Wins!")
            break

        if not get_empty_cells(board):
            print("Draw!")
            break

        current_player = 'X' if current_player == 'O' else 'O'

main()


# Task6 When planning a road trip, we are trying to minimize our costs in many different areas - gas,
# time, overnight stays, traffic costs, etc. Calculating these costs can take a lot of effort and
# time, Pathfinding algorithms are one of the utility-based agent and classical graph problem.
# The Shortest Path algorithm is an algorithm that calculates a path between two nodes in a
# weighted graph such as the sum of the values on the edges that form a path is minimized.
# Starting from the source node, the algorithm looks up the weights on the (out-)going (in
# weighted graphs) edges. It chooses the edge which, summed to the previous total sum,
# gives the lowest result. The algorithm runs through every node up until the destination point.
# Results are a path and the total sum of the shortest path. Your task is to write a program of
# utility-based agent that find the best route (minimum Distance, shortest path) from source to
# destination node.

# In[14]:


import heapq

def calculate_shortest_path(graph, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return cost, path
            for next_node, distance in graph[node]:
                if next_node not in visited:
                    heapq.heappush(queue, (cost + distance, next_node, path))
    return float("inf"), []

graph = {
    "Fast NU": [("Karsaz", 15), ("Korangi", 20)],
    "Karsaz": [("Fast NU", 15), ("Gulshen", 8), ("Sadar", 10)],
    "Gulshen": [("Karsaz", 8), ("Korangi", 30), ("Sadar", 20)],
    "Sadar": [("Karsaz", 10), ("Gulshen", 20)],
    "Korangi": [("Fast NU", 20), ("Gulshen", 30)]
}

shortest_distance, shortest_path = calculate_shortest_path(graph, "Fast NU", "Sadar")
print(shortest_distance, shortest_path)


# Task#07 Learning Agent
# Any agent, model, utility or goal-based agent can be transformed into learning agent. Few
# examples are snake game, Pac man, Self-driving cars, Ad recommendation system.
# Read the below article and write few sentences that how you can transform task 04, 05 and
# 06 into learning agent (NO code necessary just reason).
# https://vitalflux.com/reinforcement-learning-real-world-examples.

# Transforming tasks 04 (vacuum cleaner), 05 (tic tac toe), and 06 (pathfinding) into learning agents involves using machine learning to enhance performance. For the vacuum, reinforcement learning could optimize cleaning paths. In tic tac toe, a deep learning model can learn winning strategies. For pathfinding, machine learning algorithms could refine route efficiency. These approaches enable agents to adapt and improve through experience, making decisions more effectively.

# In[ ]:




