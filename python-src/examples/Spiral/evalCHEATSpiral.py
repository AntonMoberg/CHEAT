import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os
from config import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from population import *
from visualize import *
from math import log
import tqdm

def evalFunc(genome, inputs, targets):
    fitness = 0.
    for i,t in zip(inputs, targets):
        o = genome.network.activate(i)
        # if t[0] == 1.0 and o[0] > 0.5:
        #     fitness += 1
        # elif t[0] == 1.0 and o[0] < 0.5:
        #     pass
        # elif t[0] == 0.0 and o[0] > 0.5:
        #     pass
        # elif t[0] == 0.0 and o[0] < 0.5:
        #     fitness += 1
        # else:
        #     raise ValueError('Should not be able to happen...')
        fitness += -sum(np.array(t)* log( max( np.array(o), 1e-15 ) ) + ( 1 - np.array(t) ) * log( max( 1 - np.array(o), 1e-15 ) ) )
        # fitness += int(abs(round(t[0]) - round(o[0])))
    genome.fitness = fitness/len(inputs)
    # genome.fitness = fitness

def spirals(n_points, noise=.5, nturns = 2):
    """
     Returns the two spirals dataset.
    """
    nturn = nturns
    n = np.sqrt(np.random.rand(n_points,1)) * nturn * (2*np.pi)
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))


if __name__ == "__main__":
    nturns = 2
    d,t = spirals(410, 0, nturns)
    t = [[i] for i in t]

    config = Config()

    pop = Population(config, d, t, evalFunc)

    pop.runCHEAT()
    with open(os.path.join('pop.pkl'), 'wb') as f:
        pickle.dump(pop,f)
    # with open(os.path.join('pop.pkl'), 'rb') as f:
    #     pop = pickle.load(f)

    print(pop.bestFitness[-1][0])

    visualize = Visualize(pop, 'Spiral {} turns'.format(nturns) + '\n' + 'Fully Connected, Adaptive Growth: OFF')

    visualize.bestNetwork()
    visualize.fitnessBest()
    visualize.fitnessPop()
    visualize.sizePop()
    visualize.boundryPlot(res = (500,500))