#!/usr/bin/env python
# coding: utf-8

# Task 1
# Implement Game Search Algorithm to solve the tic-tac-toe problem mentioned below.

# In[17]:


#please note that i have added comments for my better understanding of this code as i was understanding and implementing 
# it from a tutorial step by step. in the rest of the questions i didnt added comments as i able to understand them through ur 
#lab manuals, however this was a tough one hence added comments as well.

def game_board(board):
    for i, row in enumerate(board):
        print('|'.join(row))
        if i < len(board) - 1:
            print('-' * 5)
#3x3 board for tic tac toe 
board = [[' 'for _ in range(3)] for _ in range(3)]
game_board(board)

#game environment
def check_win(board, player):
    #check rows 
    for row in board:
        if all(cell == player for cell in row):
            return True
        
        #check columns
    for column in range(3):
        if all(board[row][column] == player for row in range(3)):
            return True
        
        #check diagnols
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False


#if the game gets drawn
def draw(board):
    return all(cell != ' ' for row in board for cell in row)

#player moves against the AI Agent
def player_move(board):
    while True:
        try:
            row = int(input("enter row (0,1,2) "))
            column = int(input("enter column (0,1,2) "))
            if 0 <= row <= 2 and 0 <= column <= 2 and board[row][column] == ' ':
                return row, column
            else:
                print("Invalid move, try again.")
        except ValueError:
            print("Invalid input, enter numbers from 0-2.")

#AI moves against the player
def ai_move(board, max_depth):
    best_eval = -float('inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                eval = minmax(board, 0, False, max_depth)
                board[i][j] = ' '
                if eval > best_eval:
                    best_eval = eval
                    best_move = (i, j)
    return best_move

#building Minmax AI player
def minmax(board, depth, is_maximizing, max_depth):
    if check_win(board, 'X'):
        return -1
    elif check_win(board, 'O'):
        return 1
    elif draw(board) or depth == max_depth:
        return 0
    if is_maximizing:
        max_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minmax(board, depth + 1, False, max_depth)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minmax(board, depth + 1, True, max_depth)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
        return min_eval

#playing against the A1 player
def main():
    
    #reset board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    player_turn = True
    while True:
        game_board(board)
        if player_turn:
            print("\nyour turn.")
            row, column = player_move(board)
            board[row][column] = 'X'
        else:
            input("\npress enter for AI player to go")
            move = ai_move(board, 0)
            if move:
                row, column = move
                board[row][column] = 'O'
        if check_win(board, 'X'):
            game_board(board)
            print("you win")
            break
        elif check_win(board, 'O'):
            game_board(board)
            print("AI wins")
            break
        elif draw(board):
            game_board(board)
            print("It's a draw")
            break
        player_turn = not player_turn

if __name__ == "__main__":
    main()


# Task 2
# Solve the below tree by using alpha-beta pruning method.

# In[1]:


maximum = 1000
minimum = -1000

def alpha_beta(d, node, maxP, v, A, B):
    if d == 3:
        return v[node]

    if maxP:
        best = minimum
        for i in range(2):
            val = alpha_beta(d + 1, node * 2 + i, False, v, A, B)
            best = max(best, val)
            A = max(A, best)

            if B <= A:
                break
        return best

    else:
        best = maximum
        for i in range(2):
            val = alpha_beta(d+1, node*2+i, True, v,A,B)
            best = min(best, val)
            B = min(B, best)

            if B <= A:
                break
        return best

scr = []
x = int(input("enter total number of leaf nodes "))
for i in range(x):
    y = int(input("enter node value "))
    scr.append(y)

d = int(input("enter current depth value "))
node = int(input("enter node value "))

print("Optimal value is ",alpha_beta(d,node,True,scr, minimum, maximum))


# Task 3 Implement N-Queen Problem in Constraint Satisfaction Problem.

# In[4]:


def is_valid_assignment(assignment, row, col):
    for previous_row, previous_col in enumerate(assignment[:row]):
        if previous_col == col:
            return False
        if abs(previous_row - row) == abs(previous_col - col):
            return False
    return True

def n_queens_csp(n):
    assignment = [-1] * n

    def backtracking(row):
        if row == n:
            return True
        for col in range(n):
            if is_valid_assignment(assignment, row, col):
                assignment[row] = col
                if backtracking(row+1):
                    return True
                assignment[row] = -1
        return False

    if backtracking(0):
        board = [["." for _ in range(n)] for _ in range(n)]
        for row,col in enumerate(assignment):
            board[row][col] = "1"
        return board
    else:
        return None

n = 8
board = n_queens_csp(n)
if board:
    for row in board:
        print(' '.join(row))
else:
    print("solution doesn't exist")


# Task 4
# Solve Below Cryptarithmetic Problem

# In[6]:


from itertools import permutations

def cryptarithmetic():
    for perm in permutations(range(10), 7):
        b,a,s,e,l,m,g = perm
        if b == 0 or g == 0:
            continue
            
        Base = b*1000 + a*100 + s*10 + e
        ball = b*1000 + a*100 + l*10 + l
        Games = g*10000 + a*1000 + m*100 + e*10 + s
        if Base + ball == Games:
            return {'B': b, 'A': a, 'S': s, 'E': e, 'L': l, 'M': m, 'G': g}

solution = cryptarithmetic()
if solution:
    print("solution exist")
    for letter, digit in solution.items():
        print(f"{letter}:{digit}")
else:
    print("solution doesn't exist")


# MinMax Algorithm

# In[1]:


import math

def func_MinMax(cd, node, maxt, scr, td):
  if cd==td:
    return scr[node]

  if maxt:
    return max(func_MinMax(cd+1, node*2, False, scr, td), func_MinMax(cd+1, node*2+1, False, scr, td))

  else:
    return min(func_MinMax(cd+1, node*2, True, scr, td), func_MinMax(cd+1, node*2+1, True, scr, td))


scr = []

x = int(input("Enter the Total Number of Leaves: "))

for i in range(x):
  y = int(input("Enter the Leaf Value: "))
  scr.append(y)

td = math.log(len(scr), 2)
cd = int(input("Current Depth Value: "))
nodev = int(input("Enter the Node Value: "))
maxt= True
print("The Answer is: ", end="")
answer = func_MinMax(cd,nodev, maxt, scr, td)
print(answer)


# In[ ]:




