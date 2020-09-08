import matplotlib.pyplot as plt
import numpy as np
import math
import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from matplotlib import pyplot
from math import cos, sin, atan

class Neuron():
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx
    
    def draw(self, neuronRadius):
        circle = pyplot.Circle((self.x, self.y), radius=neuronRadius, fill=False)
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, neuronsInLayer, numberOfNeuronsInWidestLayer, idx):
        self.idx = idx
        self.verticalDistanceBetweenLayers = 12
        self.horizontalDistanceBetweenNeurons = 6
        self.neuronRadius = 0.75
        self.numberOfNeuronsInWidestLayer = numberOfNeuronsInWidestLayer
        self.network = network
        self.y = self.__calculateLayerYPosition()
        self.neurons = self.__initialiseNeurons(neuronsInLayer)
    
    def __initialiseNeurons(self, neuronsInLayer):
        neurons = []
        x = self.__calculateLeftMarginSoLayerIsCentered(len(neuronsInLayer))
        for neuronIdx in neuronsInLayer:
            neuron = Neuron(x, self.y, neuronIdx)
            neurons.append(neuron)
            x += self.horizontalDistanceBetweenNeurons
        return neurons
    
    def __calculateLeftMarginSoLayerIsCentered(self, numberOfNeurons):
        return self.horizontalDistanceBetweenNeurons * (self.numberOfNeuronsInWidestLayer - numberOfNeurons) / 2
    
    def __calculateLayerYPosition(self):
        return self.idx * self.verticalDistanceBetweenLayers
    
    def __lineBetweenTwoNeurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuronRadius * sin(angle)
        y_adjustment = self.neuronRadius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)
    
    def draw(self):
        for neuron in self.neurons:
            neuron.draw(self.neuronRadius)
            if self.idx > 0:
                for layer in self.network.layers:
                    for neuronPrev in layer.neurons:
                        if neuronPrev.idx in list(self.network.genome.nodes[neuron.idx].inputs.keys()):
                            self.__lineBetweenTwoNeurons(neuron, neuronPrev)
        
        # write Text
        x_text = self.numberOfNeuronsInWidestLayer * self.horizontalDistanceBetweenNeurons
        if self.idx == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif self.idx == len(self.network.genome.layers)-1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(self.idx), fontsize = 12)

class NeuralNetwork():
    def __init__(self, numberOfNeuronsInWidestLayer, genome):
        self.numberOfNeuronsInWiderstLayer = numberOfNeuronsInWidestLayer
        self.layers = []
        self.layertype = 0
        self.genome = genome

    def addLayer(self, neurons):
        layer = Layer(self, neurons, self.numberOfNeuronsInWiderstLayer, len(self.layers))
        self.layers.append(layer)
    
    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            layer.draw()
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()

class DrawNN():
    def __init__(self, genome):
        self.neuralNet = [genome.layers.get(key) for key in sorted(list(genome.layers.keys()))]
        self.genome = genome

    def draw(self):
        widestLayer = max([len(layer) for layer in self.neuralNet])
        network = NeuralNetwork(widestLayer, self.genome)
        for layer in self.neuralNet:
            network.addLayer(layer)
        network.draw()


class Visualize(object):
    def __init__(self, pop, customEnding = None):
        self.titleEnding = customEnding
        self.pop = pop 

    def fitnessBest(self):
        pop = self.pop
        plt.plot(np.linspace(0,len(pop.bestFitness)-1, len(pop.bestFitness)), [g[0] for g in pop.bestFitness])
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Cross-Entropy)')
        plt.title('Fitness of best individual in population - ' + self.titleEnding)
        plt.grid()
        plt.show()

    def fitnessPop(self):
        pop = self.pop
        for i in range(len(pop.populationFitness[0])):
            plt.plot(np.linspace(0,len(pop.bestFitness)-1, len(pop.bestFitness)), [gen[i] for gen in pop.populationFitness])
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Cross-Entropy)')
        plt.title('Fitness of each individual in population - ' + self.titleEnding)
        plt.grid()
        plt.show()

    def avgSizePop(self):
        pop = self.pop
        plt.plot(np.linspace(0,len(pop.avgSize)-1, len(pop.avgSize)), pop.avgSize, label='Population Average')
    
    def bestSizePop(self):
        pop = self.pop
        plt.plot(np.linspace(0,len(pop.bestFitness)-1, len(pop.bestFitness)), [len(g[1].nodes) for g in pop.bestFitness], label='Best Individual')
    
    def sizePop(self):
        self.avgSizePop()
        self.bestSizePop()
        plt.xlabel('Generation')
        plt.ylabel('Size (Nodes)')
        plt.title('Network size in population - ' + self.titleEnding)
        plt.legend()
        plt.grid()
        plt.show()
    
    def boundryPlot(self, res = (100,100)):
        pop = self.pop
        g = pop.bestFitness[-1][1]

        x = [i[0] for i in pop.inputs]
        y = [i[1] for i in pop.inputs]
        xMax, xMin = max(x), min(x)
        yMax, yMin = max(y), min(y)

        xArr = np.linspace(math.floor(xMin),math.ceil(xMax), res[0])
        yArr = np.linspace(math.floor(yMin),math.ceil(yMax), res[1])

        im = np.empty(shape = (len(xArr), len(yArr)))

        fig, ax = plt.subplots()

        bar = tqdm.tqdm(desc = 'Pixel', total = len(xArr)*len(yArr))
        for i,x in enumerate(xArr):
            for j,y in enumerate(yArr):
                o = g.network.activate([x,y])[0]
                if o < 0.5:
                    im[j][i] = o
                elif o > 0.5:
                    im[j][i] = o
                bar.update(1)

        img = ax.imshow(im, extent = [xMin,xMax,yMin,yMax], cmap = 'seismic', alpha = 0.75, aspect = 'auto', origin = 'lower')
        plt.colorbar(img, ax=ax)
        # ax.autoscale(False)

        for i,target in zip(pop.inputs, pop.targets):
            if target == [0.0]:
                color = 'b'
            else:
                color = 'r'
            
            ax.scatter(i[0],i[1], c = color)

        plt.title('Boundary plot - ' + self.titleEnding)
        plt.show()
    
    def bestNetwork(self):
        g = self.pop.bestFitness[-1][1]

        network = DrawNN(g)
        network.draw()
        
