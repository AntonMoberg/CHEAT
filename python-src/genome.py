import numpy as np
import math
import copy
from random import choice, sample
from nn import *

class Node(object):
    
    def __init__(self, config):
        self.layer_idx = None
        self.bias = None
        self.func = None
        self.inputs = {}
        self.config = config

    #Defines sigmoid activation function
    def sigmoid(self, x, gradient = False):
        if gradient == True:
            return x*(1-x)
        else:
            if x < 0:
                return math.exp(x)/(1 + math.exp(x))
            else:
                return 1/(1+math.exp(-x))
    
    #Defines tanh activation function
    def tanh(self, x):
        return math.tanh(x)
    
    #Defines which activation fucntion to use
    def chooseFunc(self):
        config = self.config
        if 'sigmoid' in config.activation_func:
            self.func = self.sigmoid
        elif 'tanh' in config.activation_func:
            self.func = self.tanh
        else:
            raise NotImplementedError("Allowed choices of activation functions are: ['sigmoid', 'tanh']")

    #Configues the node as an input node
    def configInput(self):
        #Define parameters of node object
        self.layer_idx = 0
        self.bias = np.array([1.,0.])
        self.inputs = {}
        self.chooseFunc()

    #configues the node as a hidden node
    def configHidden(self, inputs):
        #Define parameters of node object
        self.layer_idx = 1
        self.bias = np.array([self.randBias(),0.])
        for node in inputs:
            self.inputs[node] = np.array([self.randWeight(),0.])
        self.chooseFunc()
    
    #Configues the node as an output node
    def configOutput(self, inputs):
        self.layer_idx = 2
        self.bias = np.array([self.randBias(),0.])
        for node in inputs:
            self.inputs[node] = np.array([self.randWeight(),0.])
        self.chooseFunc()
    
    def configAddedNode(self, n1, n2, genome):
        self.layer_idx = (genome.nodes[n1].layer_idx + genome.nodes[n2].layer_idx) / 2
        self.bias = np.array([self.randBias(),0.])
        self.inputs[n2] = np.array([1.,0.])
        self.chooseFunc()
    
    def configAddedFullyConnected(self,genome,lb,l):
        self.layer_idx = l
        self.bias = np.array([self.randBias(),0.])
        for node in genome.layers[lb]:
            self.inputs[node] = np.array([1.,0.])
        self.chooseFunc()

    #Generates a random bias
    def randBias(self):
        config = self.config
        #Generate random bias with given sigma and mean
        bias = np.random.normal(config.bias_val_mean, config.bias_val_sigma)
        #Adjust if too high/low
        if abs(bias) > config.bias_val_max:
            bias = bias*(config.bias_val_max/abs(bias))
        return bias
    
    #Generates a random weight
    def randWeight(self):
        config = self.config
        #Generate random weight with given sigma and mean
        weight = np.random.normal(config.weight_val_mean, config.weight_val_sigma)
        #Adjust if too high/low
        if abs(weight) > config.weight_val_max:
            weight = weight*(config.weight_val_max/abs(weight))
        return weight

class Genome(object):

    def __init__(self, config, innovNumber):

        #Fitness of genome (int/float)
        self.fitness = None
        #Nodes in genome (dict of values or dict of object)
        #TODO: Decide (dict of values or dict of object)?
        self.nodes = {}
        # self.layers = {}
        #Config parameters for genome
        self.config = config
        #Defines which nodes are in which layers
        self.layers = {}
        #Initiates a global counter for historical markings
        self.innov_number = innovNumber
        #Creates the FeedForward network associated with the genome
        self.network = FeedForward(self)

    #Defines and constructs a new genome
    def configNewGenome(self):
        config = self.config

        #Create input nodes
        for i in range(config.num_in):
            #Initialize Node object
            node = Node(config)
            #Define parameters of node object
            node.configInput()
            #Add nodes to genome
            self.nodes[i] = node
            
        
        #Configure hidden layer
        for i in range(config.num_hidden):
            #Initialize Node Object
            node = Node(config)
            #Define parameters of node object
            node.configHidden([i for i in range(config.num_in)])
            #Add nodes to genome
            self.nodes[config.num_in + config.num_out + i] = node


        #Configure output layer
        for i in range(config.num_out):
            #Initialize Node object
            node = Node(config)
            #Define parameters of node object
            if config.num_hidden != 0:
                node.configOutput([i + config.num_in + config.num_out for i in range(config.num_hidden)])
            else:
                node.configOutput([i for i in range(config.num_in)])
            self.nodes[config.num_in] = node
        
        #Update layers
        for (i, n) in self.nodes.items():
            try:
                self.layers[n.layer_idx].append(i)
            except:
                self.layers[n.layer_idx] = [i]

    def configCrossoverGenome(self, genome1, genome2):
        assert isinstance(genome1.fitness, (int,float))
        assert isinstance(genome1.fitness, (int,float))

        #Chose parent depending on what the fitness goal is (maximize or minimize)
        if self.config.fitness_goal == 'max':
            if genome1.fitness > genome2.fitness:
                parent1, parent2 = genome1, genome2
            else:
                parent1, parent2 = genome2, genome1
        else:
            if genome1.fitness < genome2.fitness:
                parent1, parent2 = genome1, genome2
            else:
                parent1, parent2 = genome2, genome1
                
        self.nodes = {**copy.deepcopy(parent1.nodes), **copy.deepcopy(parent2.nodes)}

        for key in self.nodes.keys():
            self.nodes[key].inputs = {}
            if parent1.nodes.get(key) is None:
                for nodeIn in parent2.nodes[key].inputs.keys():
                    self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent2.nodes[key].inputs[nodeIn])

            elif parent2.nodes.get(key) is None:
                for nodeIn in parent1.nodes[key].inputs.keys():
                    self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent1.nodes[key].inputs[nodeIn])
            
            else:
                nodesIn = list(set().union(parent1.nodes[key].inputs.keys(), parent2.nodes[key].inputs.keys()))
                for nodeIn in nodesIn:
                    if parent1.nodes[key].inputs.get(nodeIn) is None:
                        self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent2.nodes[key].inputs[nodeIn])

                    elif parent2.nodes[key].inputs.get(nodeIn) is None:
                        self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent1.nodes[key].inputs[nodeIn])
                    
                    else:
                        if np.random.uniform() > 0.5:
                            self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent1.nodes[key].inputs[nodeIn])
                        else:
                            self.nodes[key].inputs[nodeIn] = copy.deepcopy(parent2.nodes[key].inputs[nodeIn])
        
        #Update Layers
        for (i, n) in self.nodes.items():
            try:
                self.layers[n.layer_idx].append(i)
            except:
                self.layers[n.layer_idx] = [i]

    
    def mutateWeights(self):
        mp = self.config.mutate_weight_prob
        mw = self.config.mutate_weight_width
        for node in self.nodes.values():
            if node.layer_idx == 0:
                pass
            else:
                for weight in node.inputs.values():
                    if np.random.uniform() < mp:
                        weight[0] += np.random.normal(scale = mw)
                    else:
                        pass
    
    def randomizeWeights(self):
        for node in self.nodes.values():
            if node.layer_idx == 0:
                pass
            else:
                node.bias = np.array([node.randBias(), 0.])
                for nodeIn in node.inputs:
                    node.inputs[nodeIn] = np.array([node.randWeight(),0.])

    
    def mutateTopology(self):
        if self.config.fullyConnectedLayers == True:
            addConnProb = 0
        else:
            addConnProb = self.config.mutate_add_conn

        if self.config.adaptiveGrowth == True:
            nGrowth = 0
            pS = self.config.adaptiveGrowthParam*(len(self.nodes)+sum([len(self.nodes[n].inputs) for n in self.nodes]))

            while nGrowth < pS:
                if np.random.uniform() < self.config.mutate_add_node:
                    self.addNode()
                    nGrowth += 1
                else:
                    pass
                
                if np.random.uniform() < addConnProb:
                    self.addLink()
                    nGrowth += 1
                else:
                    pass
        
        else:
            if np.random.uniform() < self.config.mutate_add_node:
                self.addNode()
            else:
                pass
            
            if np.random.uniform() < self.config.mutate_add_conn:
                self.addLink()
            else:
                pass
    
    def addNode(self):
        #Create new node object
        newNode = Node(self.config)

        if self.config.fullyConnectedLayers == False:
            #Select in between which two nodes the new node shall be inserted
            selectedNode1Idx = choice(list(self.nodes.keys())[self.config.num_in:])
            selectednode2Idx = choice(list(self.nodes[selectedNode1Idx].inputs.keys()))

            #Configure the parameters of the new node
            newNode.configAddedNode(selectedNode1Idx, selectednode2Idx, self)
            #Assign an index to the new node and insert into the genome
            nodeIdx = self.innov_number.next()
            self.nodes[nodeIdx] = newNode

            #Remove selectedNode2 from selectedNode1 input and add newNode with same weight
            weight = self.nodes[selectedNode1Idx].inputs[selectednode2Idx]
            del self.nodes[selectedNode1Idx].inputs[selectednode2Idx]
            self.nodes[selectedNode1Idx].inputs[nodeIdx] = weight

            #Update layers with new node
            try:
                self.layers[newNode.layer_idx].append(nodeIdx)
            except:
                self.layers[newNode.layer_idx] = [nodeIdx]
        
        elif self.config.fullyConnectedLayers == True:

            if np.random.uniform() < self.config.depthBreadthRatio:

                if len(list(self.layers.keys())[1:-1]) == 0:
                    l1,l2 = list(self.layers.keys())[0], list(self.layers.keys())[-1]
                    selectedLayer = (list(self.layers.keys())[0] + list(self.layers.keys())[-1])/2

                else:
                    l2Idx = sorted(list(self.layers.keys())).index(choice(sorted(list(self.layers.keys()))[1:]))
                    l1 = sorted(list(self.layers.keys()))[l2Idx-1]
                    l2 = sorted(list(self.layers.keys()))[l2Idx]
                    selectedLayer = (l1+l2)/2
                    

                newNode.configAddedFullyConnected(self, l1, selectedLayer)
                nodeIdx = self.innov_number.next()
                self.nodes[nodeIdx] = newNode

                for node in self.layers[l2]:
                    self.nodes[node].inputs = {}
                    self.nodes[node].inputs[nodeIdx] = np.array([newNode.randWeight(),0.])
            
            else:

                if len(list(self.layers.keys())[1:-1]) == 0:
                    l1,l2 = list(self.layers.keys())[0], list(self.layers.keys())[-1]
                    selectedLayer = (list(self.layers.keys())[0] + list(self.layers.keys())[-1])/2
                    for node in self.layers[l2]:
                        self.nodes[node].inputs = {}

                else:
                    selectedLayerIdx = sorted(list(self.layers.keys())).index(choice(sorted(list(self.layers.keys()))[1:-1]))
                    l1,l2 = sorted(list(self.layers.keys()))[selectedLayerIdx-1], sorted(list(self.layers.keys()))[selectedLayerIdx+1]
                    selectedLayer = sorted(list(self.layers.keys()))[selectedLayerIdx]

                newNode.configAddedFullyConnected(self,l1,selectedLayer)
                nodeIdx = self.innov_number.next()
                self.nodes[nodeIdx] = newNode

                for node in self.layers[l2]:
                    self.nodes[node].inputs[nodeIdx] = np.array([newNode.randWeight(),0.])

            try:
                self.layers[newNode.layer_idx].append(nodeIdx)
            except:
                self.layers[newNode.layer_idx] = [nodeIdx]

        else:
            raise ValueError('fullyConnectedLayers need to be either True or False, not {}'.format(self.config.fullyConnectedLayers))

    
    def addLink(self):
        #Select two layers to chose nodes from
        selectedLayers = sorted(sample(self.layers.keys(), 2))
        #Chose one node from respective layer to connect
        n1 = choice(self.layers[selectedLayers[0]])
        n2 = choice(self.layers[selectedLayers[1]])

        #If connection already exist, do nothing else add connection with random weight
        if isinstance(self.nodes[n2].inputs.get(n1), np.ndarray):
            pass
        else:
            self.nodes[n2].inputs[n1] = np.array([self.nodes[n2].randWeight(), 0])





#################TESTS####################

# from config import *
# from nn import *

# config = Config()

# g = Genome(config)
# g.configNewGenome()
# net = FeedForward(g)
# print(net.activate([1,1]))
# for i in range(1000):
#     net.backProp([0])
#     print(net.activate([1,1]))
