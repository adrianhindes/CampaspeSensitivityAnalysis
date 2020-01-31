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
#Get set of labels used in input and output columns
inputTerms = set(l1)
outputTerms = set(outputCols)

#Specify model results to do SA on
filterTerms = ["goulburn_allocation"]

avgYLabels = [x for x in outputCols.tolist() if any(s in x for s in filterTerms)]

# Select model output to perform sensitivity analysis across
# test case
y0Label = avgYLabels[0]
dfY = outputs[y0Label]
dfY = dfY.fillna(0)
Y0 = dfY.to_numpy()


# k = no. model inputs
# n = sample size ie. number of model evaluations


inputsNum = inputs._get_numeric_data()
noConsts = inputsNum.loc[:, inputsNum.apply(pd.Series.nunique) != 1]

findBounds = lambda c: [min(c), max(c)]

bounds = []
for col in noConsts:
    bounds.append(findBounds(noConsts[col]))



def plot_convergence(x,y):
    """
    Visualize the convergence of the sensitivity indices
    takes two arguments : x,y input and model output samples
    return plot of sensitivity indices wrt number of samples
    """
    try:
        ninput = x.shape[1]
    except (ValueError, IndexError):
        ninput = x.size
    
    
    try:
        noutput = y.shape[1]
    except (ValueError, IndexError):
        noutput = y.size

    nsamples = x.shape[0]
    trials = (nsamples-30)//10
    all_si_c = np.zeros((trials, ninput))  # ninput
    for i in range(30,nsamples,10):
        modProblem = {
            'num_vars': ninput,
            'names': inputLabels,
            'bounds': bounds
        }
        
        # all_si_c[(i-30)//10,:] = rbdfast(y[:i],x=x[:i,:])[1]
        res = rbd_fast.analyze(modProblem, x[:i], y[:i], M=10)

        all_si_c[((i-30)//10)-1, ] = res['S1']
        
        
    plt.plot([all_si_c[i].mean() for i in range(trials)])
    # for i in range(trials):
    #     plt.plot(all_si_c[i].mean(),'k-')
    plt.show()
    return



# inputsNum = inputs._get_numeric_data()
inputLabels = noConsts.columns.tolist()
inputLabels = list(map(delUnder, inputLabels))

X = noConsts.values
plot_convergence(X, Y0)


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

Si = rbd_fast.analyze(problem, X, Y0, M=10, print_to_console=True)

Si.plot()

