#!/usr/bin/env python
# coding: utf-8

# # K22-4005 Mujtaba Saqib BAI-4A

# In[4]:


import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Task 1: Problem: Suppose you have a fair coin and an unfair coin. The probability of getting heads when flipping the
# fair coin is 0.5, while the probability of getting heads when flipping the unfair coin is 0.8. You randomly select one of
# the two coins and flip it. What is the probability that you selected the fair coin given that you got heads?

# In[5]:


def naive_bayes(coin_A_head,Head,coin_A):
       return (coin_A_head*coin_A)/Head
    

total_coins=2
coin_A=0.5
coin_B=0.5
coin_A_head=0.5
coin_A_Tail=0.5
coin_B_head=0.8
coin_B_Tail=0.2

Head=(coin_A_head*coin_A)+(coin_B_head*coin_B)
print(naive_bayes(coin_A_head,Head,coin_A))


# In[6]:


model = BayesianModel([('coin','result')])

#defining the variables, 2 coins hence 2 variables and containing the their heads and tail probabilities and total coins
# are 2 hence 0.5 for each coin. rest done as per lab manual.

cpd1 = TabularCPD(variable='coin', variable_card=2, values=[[0.5], [0.5]],state_names={'coin': ['A', 'B']})

#head probabilities coin A 0.5 and for tail 0.5, coin b head 0.8 and tail 0.2
cpd2 = TabularCPD(variable='result', variable_card=2,values=[[0.5, 0.8], [0.5, 0.2]],evidence=['coin'], evidence_card=[2],
                  state_names={'coin':['A','B'], 'result': ['heads','tails']})

model.add_cpds(cpd1,cpd2)
assert model.check_model()

inference = VariableElimination(model)
result = inference.query(variables=['coin'], evidence={'result': 'heads'})
print(result)


# Task 2: Problem: Suppose you have a standard deck of playing cards with 52 cards. You draw one card at random.
# What is the probability that the card drawn is a heart?

# In[7]:


def hearts_probability(event_outcomes, total_cards):
    prob = event_outcomes / total_cards
    return prob

#a deck has total 52 cards and total hearts are 13
cards = 52 
hearts = 13

#function passed to calculate probability
heart_probability = hearts_probability(hearts, cards)
print("probability of heart ",str(heart_probability))


# In[8]:


get_ipython().system('pip install pomegranate')


# In[9]:


get_ipython().system('pip install pgmpy')


# Task 3: Problem: Consider a scenario where a student's performance in an exam may depend on various factors such as attendance, study time, and previous exam scores. Design a Bayesian network to represent this scenario and calculate the probability of a student passing the exam given their attendance and study time.

# In[39]:


from pomegranate import *

attendance = DiscreteDistribution({'good': 0.4, 'poor': 0.6})
study_time = DiscreteDistribution({'high': 0.3, 'medium': 0.4, 'low': 0.3})
previous_scores = DiscreteDistribution({'good': 0.3, 'Average': 0.4, 'poor': 0.3})

exam_performance = ConditionalProbabilityTable(
     [['good', 'high', 'good', 'pass', 0.35],
     ['good', 'high', 'Average', 'pass', 0.55],
     ['good', 'high', 'poor', 'pass', 0.25],
     ['good', 'medium', 'good', 'pass', 0.35],
     ['good', 'medium', 'Average', 'pass', 0.45],
     ['good', 'medium', 'poor', 'pass', 0.95],
     ['good', 'low', 'good', 'pass', 0.75],
     ['good', 'low', 'Average', 'pass', 0.65],
     ['good', 'low', 'poor', 'pass', 0.55],
     ['poor', 'high', 'good', 'pass', 0.85],
     ['poor', 'high', 'Average', 'pass', 0.75],
     ['poor', 'high', 'poor', 'pass', 0.65],
     ['poor', 'medium', 'good', 'pass', 0.75],
     ['poor', 'medium', 'Average', 'pass', 0.65],
     ['poor', 'medium', 'poor', 'pass', 0.55],
     ['poor', 'low', 'good', 'pass', 0.65],
     ['poor', 'low', 'Average', 'pass', 0.55],
     ['poor', 'low', 'poor', 'pass', 0.45]],
    [attendance, study_time, previous_scores])

s1 = State(attendance, name="attendance")
s2 = State(study_time, name="study time")
s3 = State(previous_scores, name="previous scores")
s4 = State(exam_performance, name="exam performance")

network = BayesianNetwork("student performance")
network.add_states(s1, s2, s3, s4)
network.add_edge(s1, s4)
network.add_edge(s2, s4)
network.add_edge(s3, s4)
network.bake()

beliefs = network.predict_proba({'attendance':'good','study time':'high'})
#passing with good attendance and high study time
print(beliefs[3].parameters[0]['pass'])

beliefs_poor_attendance_high_study = network.predict_proba({'attendance':'poor','study time':'high'})
#passing with poor attendance and high study time
print(beliefs_poor_attendance_high_study[3].parameters[0]['pass'])


# Task 4: Problem: Suppose you have two fair six-sided dice. You roll both dice simultaneously. What is the probability
# that the sum of the numbers rolled is 7?

# In[ ]:


def probability(total_outcome,sum_of_7):
    prob=sum_of_7/total_outcome
    return prob

#2 dices hence 6 possibility each so 6*6=36
total_outcome=36

#(1,6),(6,1),(2,5),(5,2),(3,4),(4,3)
sum_of_7=6

print(round(probability(total_outcome,sum_of_7),3))


# Suppose we want to model a student's performance in a class based on two factors: the amount of time they spend
# studying and whether or not they have access to tutoring.
# Bayesian Network Design:
# We can design a Bayesian network with two nodes:
# ● StudyTime: representing the amount of time spent studying (low, medium, high).
# ● Tutoring: representing whether the student has access to tutoring (yes, no).
# ● Performance: representing the student's performance (poor, average, good).
# Dependencies:
# ● StudyTime and Tutoring both influence Performance.
# ● There is no direct dependency between StudyTime and Tutoring.
# Conditional Probability Distributions (CPDs):
# ● CPD for Performance given StudyTime and Tutoring.
# Questions/Tasks:
# 1. What is the probability that a student's performance is 'poor' given that they spend a 'medium' amount of time
# studying and have access to tutoring?
# 2. Given that a student's performance is 'good', what is the probability that they spend a 'high' amount of time
# studying and have access to tutoring?
# 3. If a student's performance is 'average', what is the probability distribution of their study time and tutoring?
# Use the Bayesian network model along with inference techniques to answer these questions.

# In[17]:


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


#as study time and tutoring both influence performance
model = BayesianModel([
    ('studytime', 'performance'),
    ('tutoring', 'performance')
])

#studytime, low medium high
cpd1= TabularCPD(variable='studytime', variable_card=3,
                            values=[[0.3], [0.5], [0.2]],
                            state_names={'studytime': ['low', 'medium', 'high']})
#tutoring, yes or no
cpd2 = TabularCPD(variable='tutoring', variable_card=2,
                          values=[[0.5], [0.5]],
                          state_names={'tutoring': ['yes', 'no']})
#performance, poor average or good
cpd3 = TabularCPD(variable='performance', variable_card=3,
                             values=[[0.2, 0.3, 0.1, 0.2, 0.05, 0.1],
                                     [0.5, 0.4, 0.4, 0.5, 0.2, 0.4], 
                                     [0.3, 0.3, 0.5, 0.3, 0.75, 0.5]],
                             evidence=['studytime', 'tutoring'],
                             evidence_card=[3, 2],
                             state_names={'performance': ['poor', 'average', 'good'],
                                          'studytime': ['low', 'medium', 'high'],
                                          'tutoring': ['yes', 'no']})

model.add_cpds(cpd1,cpd2,cpd3)

assert model.check_model()
inference = VariableElimination(model)

#this is for poor performance with medium study time and tutoring
result_q1 = inference.query(variables=['performance'], evidence={'studytime': 'medium', 'tutoring': 'yes'})
print(result_q1.values[0])

#this for high study time and tutoring and good performance
result_q2 = inference.query(variables=['studytime', 'tutoring'], evidence={'performance': 'good'})
high_study = result_q2.values[2] * result_q2.values[result_q2.state_names['tutoring'].index('yes')]
print(high_study)

result_q3 = inference.query(variables=['studytime', 'tutoring'], evidence={'performance': 'average'})
study_time = result_q3.values

#this for average performnace studytime
print(study_time[0]) 
#this for average performnace tutoring
print(study_time[1]) 


# In[ ]:




