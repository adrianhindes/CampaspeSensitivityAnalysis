# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:19:29 2020

@author: Adrian Hindes
Calculate main effect indices of Campaspe integrated model
"""
import numpy as np
import pandas as pd
import re
from SALib.analyze import rbd_fast
import matplotlib.pyplot as plt

inputs = pd.read_csv("data/all_scenario_inputs.csv")
inputs.drop(inputs.columns[[0]], axis=1, inplace=True)

inputsNum = inputs._get_numeric_data()

outputs = pd.read_csv("data/all_scenario_outputs.csv")
outputs.drop(outputs.columns[[0]], axis=1, inplace=True)

inputCols = inputs.columns
outputCols = outputs.columns

# Unnecessary
def underSplit(text):
    newText = re.split('_+', text)
    return newText


#Get important terms from column labels to figure out
    # which inputs and outputs to do sensitivity analysis on
flatten = lambda l: [item for sublist in l for item in sublist]
delUnder = lambda t: t.replace('_','')

#Delete underscores
l0 = inputCols.tolist()
l1 = list(map(delUnder, l0))
inputTerms = set(l1)
outputTerms = set(outputCols)

#Specify model results to do SA on
avgYLabels = [x for x in outputCols.tolist() if 'Avg.' in x]

# Select model output to perform sensitivity analysis across
# test case
y0Label = avgYLabels[0]
dfY = outputs[y0Label]
dfY = dfY.fillna(0)
Y0 = dfY.to_numpy()


# k = no. model inputs
# n = sample size ie. number of model evaluations



filterWords = 'gw'

inputLabels = [t for t in inputCols.tolist() if filterWords in t]
dfX = inputs[inputLabels]
X = dfX.to_numpy()

findBounds = lambda c: [min(c), max(c)]

bounds = []
for col in X.T:
    bounds.append(findBounds(col))
    
n, k = X.shape

if len(Y0) == n: 
    print("Inputs & Outputs agree in array length " + str(n))
else: print("Inputs & Outputs do not agree in array length")

#Setup problem dictionary for SALib
problem = {
        'num_vars': k,
        'names': inputLabels,
        'bounds': bounds
        }

Si = rbd_fast.analyze(problem, X, Y0, M=11, print_to_console=True)



