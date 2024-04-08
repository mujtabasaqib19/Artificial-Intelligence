#!/usr/bin/env python
# coding: utf-8

# Task# 01:
# Traveling Salesman Problem:
# Given a set of cities and distances between every pair of cities, the problem is to find the
# shortest possible route that visits every city exactly once and returns to the starting point. Like
# any problem, which can be optimized, there must be a cost function. In the context of TSP,
# total distance traveled must be reduced as much as possible.
# Consider the below matrix representing the distances (Cost) between the cities. Find theshortest
# possible route that visits every city exactly once and returns to the starting point.

# In[2]:


def uniform_cost_search_tsp(graph, start):
    queue = [(0, [start])]
    visited = set()
    min_cost = float('inf')
    min_path = []
    while queue:
        cost, path = heapq.heappop(queue)
        if path[-1] not in visited and len(path) == len(graph):
            visited.add(path[-1])
            cost += graph[path[-1]][start]
            path.append(start)
            if cost < min_cost:
                min_cost = cost
                min_path = path
        else:
            for i, edge_cost in enumerate(graph[path[-1]]):
                if i not in path:
                    heapq.heappush(queue, (cost + edge_cost, path + [i]))
    return min_cost, min_path

graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

start_node = 0
uniform_cost_search_tsp(graph, start_node)


# Task#2: Implement DFS on graph and tree.

# In[3]:


from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUtil(self, v, visited):
        visited[v] = True
        print(v, end=' ')

        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self.DFSUtil(neighbour, visited)

    def DFS(self, v):
        visited = [False] * (max(self.graph) + 1)
        self.DFSUtil(v, visited)

g = Graph(4)
g.addEdge(1, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 1)
g.addEdge(2, 4)
g.addEdge(3, 1)
g.addEdge(3, 4)
g.addEdge(4, 1)
g.addEdge(4, 2)
g.addEdge(4, 3)

print("DFS starting from vertex 1 ")
g.DFS(1)


# Task # 03: Write a program to solve the 8-puzzle problem using the DFS and BFS search algorithm

# In[7]:


class Puzzle:
    def __init__(self, start_state):
        self.board = start_state
        self.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.visited = set()
        self.stack = deque([self.board])
        self.parent = {self.__to_string(self.board): None}
        
    def __to_string(self, s):
        return ''.join(str(d) for row in s for d in row)
    
    def __is_solvable(self, s):
        num_inversions = 0
        s_list = [j for sub in s for j in sub if j != 0]
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if s_list[i] > s_list[j]:
                    num_inversions += 1
        return num_inversions % 2 == 0
    
    def __get_new_board(self, x, y, new_x, new_y, old):
        new_board = [row[:] for row in old]
        new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
        return new_board
    
    def __get_possible_moves(self, s):
        x, y = next((x, y) for x in range(3) for y in range(3) if s[x][y] == 0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_moves = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < 3 and 0 <= y + dy < 3]
        return [self.__get_new_board(x, y, new_x, new_y, s) for new_x, new_y in valid_moves]
    
    def __get_path(self, s):
        path = []
        while s:
            path.append(s)
            s = self.parent[self.__to_string(s)]
        return path[::-1]
    
    def solve(self):
        if not self.__is_solvable(self.board):
            return None
        while self.stack:
            current_board = self.stack.pop()
            self.visited.add(self.__to_string(current_board))
            if current_board == self.goal:
                return self.__get_path(current_board)
            for move in self.__get_possible_moves(current_board):
                move_str = self.__to_string(move)
                if move_str not in self.visited:
                    self.parent[move_str] = current_board
                    self.stack.append(move)
        return None

start_state = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
puzzle = Puzzle(start_state)
solution = puzzle.solve()

if solution:
    print("initial state ")
    for row in start_state:
        print(row)
    print("\ngoal state ")
    for row in solution[-1]:
        print(row)
else:
    print("no solution found")


# In[18]:


class PuzzleSolver:
    def __init__(self, initial):
        self.initial = initial
        self.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.queue = deque([(initial, [])]) 
        self.visited = set()

    def is_solvable(self, state):
        inversion_count = 0
        flat_state = [tile for row in state for tile in row if tile != 0]
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] > flat_state[j]:
                    inversion_count += 1
        return inversion_count % 2 == 0

    def solve(self):
        if not self.is_solvable(self.initial):
            return None

        while self.queue:
            current_state, path = self.queue.popleft()
            self.visited.add(self.__to_string(current_state))

            if current_state == self.goal:
                return path 

            for neighbour, action in self.get_neighbours(current_state):
                if self.__to_string(neighbour) not in self.visited:
                    self.queue.append((neighbour, path + [action]))

        return None

    def get_neighbours(self, state):
        def swap(s, i1, j1, i2, j2):
            t = [list(row) for row in s]
            t[i1][j1], t[i2][j2] = t[i2][j2], t[i1][j1]
            return t

        i, j = next((i, j) for i in range(3) for j in range(3) if state[i][j] == 0)
        moves = []
        if i>0: 
            moves.append((swap(state, i, j, i-1, j), 'Up'))
        if i<2: 
            moves.append((swap(state, i, j, i+1, j), 'Down'))
        if j>0: 
            moves.append((swap(state, i, j, i, j-1), 'Left'))
        if j<2: 
            moves.append((swap(state, i, j, i, j+1), 'Right'))
        return moves

    def __to_string(self, state):
        return ''.join(str(n) for row in state for n in row)

initial_state = [[1,2,3], [0,4,6], [7,5,8]]

solver = PuzzleSolver(initial_state)

solution_path = solver.solve()

if solution_path:
    print("initial state ")
    for row in initial_state:
        print(row)
    print("\ngoal state ")
    for row in solver.goal:
        print(row)

else:
    print("no solution exist")


# In[ ]:





# In[ ]:




