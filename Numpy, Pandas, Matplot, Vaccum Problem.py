#!/usr/bin/env python
# coding: utf-8

# # 22K-4005 Mujtaba Saqib AI lab 2

# In[33]:


import pandas as pd
import numpy as np


# • Use NumPy library and perform following tasks:
# 1. Initialize 2 arrays and add their contents.
# 2. Multiply all contents within one of the above arrays by an integer.
# 3. Reshape one of the arrays to be 2D (if existing array is 2D, make it 3D).
# 4. Convert One of the arrays to be of a different Data type.
# 5. Generate a sequence of numbers in the form of a NumPy array from 0 to 100 with gaps of 2
# numbers, for example: 0, 2, 4 ....
# 6. From 2 NumPy arrays, extract the indexes in which the elements in the 2 arrays match.

# In[34]:


#1
a=np.array([2,3,4,5,7])
b=np.array([3,5,6,7,8])

c=np.add(a,b)
print(c)


# In[35]:


#2
d=a*10
print(d)


# In[36]:


#3
arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
arr1= arr.reshape(2,3,2)
print(arr1)


# In[37]:


#4
a.astype('float64')


# In[38]:


from numpy import random


# In[39]:


#5
F=np.arange(0,101,2)
print(F)


# In[40]:


#6
e=np.array([3,4,5,6,7])
g=np.array([3,9,10,9,7])

print(np.where(e==g))


# • Use Matplotlib and perform following tasks:
# 1. Create a line chart consisting of 2 separate lines each having a label of its own and one of them styled
# in a doted manner. Also add labels to the axes.
# 2. Create a Pie Chart similar to the one given below. You can use dataset and colors of your own, but
# make sure that the structure is followed as is.

# In[41]:


import matplotlib.pyplot as plt


# In[71]:


x=[5,10,15,20]
y=[10,20,30,40]
z=[10,20,30,40]

plt.figure(figsize=(9,9))
plt.xlabel("z",fontsize=20)
plt.ylabel("x and y values", fontsize=20)
plt.title("lineplot between x and z & y and z",fontsize=20)
plt.plot(z,x,linestyle='dotted',color='darkblue',label="X line")
plt.plot(z,y,color='red',label="Y line")
plt.legend()
plt.show()


# In[70]:


expense = [500,150,200,300]
household = ["Mortage","Repairs","Food","Utilities"]
color = ["lightblue","red","darkgreen","m"]
explodes = (0.05,0.05,0.05,0.05)

plt.figure(figsize=(9,9))
plt.title("Household Expenses",fontsize=25)
plt.pie(expense,labels=household,autopct="%1.2f%%",colors=color,explode=explodes)
plt.legend(loc='upper left')
plt.legend()
plt.show()


# • Use pandas to create a new data frame consisting of 3 series and save it to a CSV file named
# ‘TestSheet.csv’. Once created, retrieve data from the same file, make changes to it and add another series
# and save the new data frame to the existing file. See the sample below for how your data should look
# like.

# In[44]:


Gym={
    "Duration":[60,60,60,45,45],
    "Pulse":[110,117,103,109,117],
    "MaxPulse":[130,145,135,175,148],
    "Calories":[409.1,479,340,282.4,406]
    }
df=pd.DataFrame(Gym)
df


# In[45]:


df.to_csv("TestSheets.csv",index=False)
df


# In[45]:


names=['Mujtaba','Sheri','Arsal','Muneeb','Abbas']
df.insert(1,"Name",names)
df


# • Perform the following tasks with NLTK.
# 1. Write a Python NLTK program to split the text sentence/paragraph into a list
# of words. Sample Below.

# In[56]:


import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
text="Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for mary and samantha at the bus station"

sentence=sent_tokenize(text)
print(f"{sentence}")


# Write a Python NLTK program to create a list of words from a given string.

# In[57]:


words=word_tokenize(text)
print(f"{words}")


# Write a Python NLTK program to tokenize words, sentence wise.

# In[58]:


sentences=sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]

for sentence in words:
    print(sentence)


# In[59]:


pip install spacy


# In[60]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[61]:


nlp = spacy.load("en_core_web_sm")


# • Use Spacy to perform the following tasks.
# 1. Create a syntactic dependency visualizer for a given sentence.
# 2. Break a given sentence into tokens each representing a single word.

# In[52]:


#1
import spacy
from spacy import displacy

doc=nlp("Babar Azam's cover drive is a masterclass in batting technique, executed with a graceful blend of balance and wrist work.")

displacy.render(doc,style='dep',jupyter=True,options={'distance':90})


# In[62]:


#2
doc=nlp("Babar Azams cover drive is a masterclass in batting technique,executed with a graceful blend of balance and wrist work.")
for token in doc:
    print(token)


# • Consider an Interactive Cognitive Environment (ICE) in which autonomous robot is
# performing cleaning task in the big room that appears to be a matrix of N * M. Each index
# referred to as a cell of the matrix is valued as dirty “D” or clean “C”. The cells which are
# occupied by the stuff in the room are blocked and valued “B”. The vacuum can move in all
# four directions (up, down, left, right), and if the cell status is D, it will clean the cell and
# change the status to “C”, if the cell status is either C, it will not enter the cell. The vacuum
# will stop working if the whole room is cleaned, i.e., the status of all the cells is either C. The
# vacuum may start cleaning the room from the first cell (0, 0) or any random location. You
# will trace the path of the vacuum and display at each step of the program. * Representthe
# location of the vacuum cleaner. Develop a Python code of the above describe scenario of
# the autonomous robot.If vacuum is in a location where it’s all neighbors (up, down, left and right) are clean (with status C)
# it will move in any one of the directions and keep searching the Dirt (D). It will stop it execution if it
# does not sense any dirt after 10 movements.
# If vacuum is in a location where it’s one more neighbor (up, down, left and right) is dirty it will move
# in any one of the directions and will return to the location when it cleans all the dirty cell of its
# neighbors. e.g., cell (0, 3) where it’s three neighbors are dirty.

# In[2]:


import random

def display(matrix, pos):
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if (i,j) == pos:
                print('*',end=' ')
            else:
                print(cell,end=' ')
        print()
    print()

def finding_neighbours(matrix, position):
    n,m = len(matrix),len(matrix[0])
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    dirty_neighbors = []
    for dx, dy in directions:
        new_x, new_y = position[0] + dx,position[1] + dy
        if 0<=new_x<n and 0 <= new_y<m and matrix[new_x][new_y]=='D':
            dirty_neighbors.append((new_x, new_y))
    return dirty_neighbors

def move_vacuum(matrix, start_position):
    n,m=len(matrix), len(matrix[0])
    position = start_position
    without_clean = 0

    while True:
        display(matrix,position)
        
        if matrix[position[0]][position[1]] == 'D':
            matrix[position[0]][position[1]] = 'C'
            without_clean =0
        else:
            without_clean +=1

        if without_clean>=10:
            print("no dirt found after 10 movements")
            break

        dirty_neighbors = finding_neighbours(matrix, position)
        if dirty_neighbors:
            position = random.choice(dirty_neighbors)
        else:
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(possible_moves)
            for dx,dy in possible_moves:
                new_x,new_y = position[0]+dx,position[1] + dy
                if 0<=new_x<n and 0<=new_y<m and matrix[new_x][new_y] != 'B':
                    position = (new_x,new_y)
                    break
            else:
                print("all cells are clean")
                break
matrix = [
    ["D", "D", "C", "B"],
    ["D", "C", "C", "D"],
    ["C", "D", "D", "C"],
    ["B", "C", "C", "D"]
]

start_position=(0,0)
move_vacuum(matrix,start_position)


# In[ ]:




