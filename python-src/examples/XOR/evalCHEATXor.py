# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from config import *
# # from nn import *
# # from genome import *
# from population import *

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os

from config import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

from population import *
from visualize import *
from math import log
import tqdm

def evalFunc(genome, inputs, targets):
    fitness = 0
    # for i,t in zip(inputs, targets):
    #     o = genome.network.activate(i)
    #     if t[0] == 1.0 and o[0] > 0.5:
    #         fitness += 1
    #     elif t[0] == 1.0 and o[0] < 0.5:
    #         pass
    #     elif t[0] == 0.0 and o[0] > 0.5:
    #         pass
    #     elif t[0] == 0.0 and o[0] < 0.5:
    #         fitness += 1
    #     else:
    #         raise ValueError('Should not be able to happen...')
    for i,t in zip(inputs, targets):
        o = genome.network.activate(i)
        fitness += -sum(np.array(t)* log( max( np.array(o), 1e-15 ) ) + ( 1 - np.array(t) ) * log( max( 1 - np.array(o), 1e-15 ) ) )
    genome.fitness = fitness


if __name__ == "__main__":
    inputs = [[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]

    config = Config()

    pop = Population(config, inputs, targets, evalFunc)

    pop.runCHEAT()
    with open(os.path.join('pop.pkl'), 'wb') as f:
        pickle.dump(pop, f)
    # with open(os.path.join('pop.pkl'), 'rb') as f:
    #     pop = pickle.load(f)

    visualize = Visualize(pop,'XOR problem')

    visualize.bestNetwork()
    visualize.fitnessBest()
    visualize.fitnessPop()
    visualize.sizePop()
    visualize.boundryPlot()
