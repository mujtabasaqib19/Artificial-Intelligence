#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Q1
#1
import numpy as np
a=np.array([1,2,3,4])
b=np.array([1,5,6,7])
c=np.add(a,b)
print(c)


# In[9]:


#2
d=np.multiply(a,3)
d


# In[10]:


#3
e=b.reshape(2,2)
e


# In[11]:


#4
f=a.astype('f')
f


# In[12]:


#5
g=np.arange(0,101,2)
g


# In[13]:


#6
np.asarray(a==b).nonzero()


# In[14]:


#Q2
#1
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7]
y=[10,20,30,40,50,60,70]
z=[20,40,60,80,100,120,140]
plt.xlabel("Number of days in week")
plt.ylabel("Dollars earned in a day")
plt.title("Economic Survey")
plt.plot(x,y,label="Ali's Earning")
plt.plot(x,z,linestyle='dotted',label="Hilal's Earning")
plt.legend(["Ali's Earning", "Hilal's Earning"])
plt.show()


# In[15]:


#Q2
#2
plt.title("Household Expenses")
label=["Mortgage","Repairs","Food","Utilities"]
sizes=[51.72,10.34,17.24,20.69]
colors=["blue","yellow","green","orange"]
explod=[0.1,0.1,0.1,0.1]
plt.pie(sizes,labels=label,colors=colors,autopct="%1.1f%%",shadow=True,explode=explod,startangle=120)
plt.figure(figsize=(8,8))


# In[16]:


#Q3
import pandas as pd
dict={
    "Duration":[60,60,60,45,45],
    "Pulse":[110,117,103,109,117],
    "Maxpulse":[130,145,135,175,148]
}
df=pd.DataFrame(dict)
df


# In[17]:


#saving dataframe to csv file
df.to_csv("TestSheet.csv",index=False)


# In[18]:


#Retrieving data from csv file
df2=pd.read_csv(r"TestSheet.csv")
df2


# In[19]:


#making changes and saving it to csv file
df2.at[2,'Pulse']=150
df2


# In[20]:


#Adding calories column to dataframe
calories=[409.1,479,340,282.4,406]
df2.insert(3,"calories",calories)
df2


# In[21]:


#Saving dataframe to csv file
df2.to_csv("TestSheet.csv",index=False)


# In[22]:


import random

class VacuumCleaner:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.room = [['D' if random.random() < 0.3 else 'C' for _ in range(cols)] for _ in range(rows)]
        self.visited = set()
        self.position = (random.randint(0, rows-1), random.randint(0, cols-1))
        self.moves_without_dirt = 0

    def display_room(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.position:
                    print("V", end=" ")  # Vacuum Cleaner
                else:
                    print(self.room[i][j], end=" ")
            print()

    def clean_cell(self, i, j):
        self.room[i][j] = 'C'
        print(f"Cleaned cell ({i}, {j})")
        self.moves_without_dirt = 0

    def move(self, direction):
        i, j = self.position
        if direction == 'up' and i > 0:
            self.position = (i-1, j)
        elif direction == 'down' and i < self.rows - 1:
            self.position = (i+1, j)
        elif direction == 'left' and j > 0:
            self.position = (i, j-1)
        elif direction == 'right' and j < self.cols - 1:
            self.position = (i, j+1)

    def sense_dirt(self):
        i, j = self.position
        return self.room[i][j] == 'D'

    def move_to_random_direction(self):
        directions = ['up', 'down', 'left', 'right']
        random.shuffle(directions)
        for direction in directions:
            self.move(direction)
            self.moves_without_dirt += 1
            if self.sense_dirt():
                return True
            if self.moves_without_dirt >= 10:
                return False
        return False

    def clean_room(self):
        while any('D' in row for row in self.room):
            self.display_room()
            i, j = self.position
            if self.sense_dirt():
                self.clean_cell(i, j)
            elif any(self.room[x][y] == 'D' for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]):
                self.move_to_random_direction()
            else:
                return  # No dirt nearby, cleaning completed
        print("Cleaning completed!")

rows, cols = 5, 5
vacuum = VacuumCleaner(rows, cols)
vacuum.clean_room()


# In[23]:


#Q5
#1
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
data="Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for mary and samantha at the bus station."
s=sent_tokenize(data)
print(s)


# In[24]:


#2
w=word_tokenize(data)
print(w)


# In[25]:


#3
sentences=sent_tokenize(data)
for words in sentences:
    w=word_tokenize(words)
    print(w)


# In[26]:


get_ipython().system('pip install spacy')


# In[27]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[34]:


nlp = spacy.load("en_core_web_sm")


# In[35]:


#Q6
#1
import spacy
from spacy import displacy


# In[36]:


#Q6
#1
data="Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."
doc = nlp(data)
displacy.render(doc, style="dep", jupyter=True, options={'distance': 90})


# In[37]:


#2
tokens = [token.text for token in doc]
print("Tokens:", tokens)


# In[ ]:




