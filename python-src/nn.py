import random
import copy
import numpy as np

class FeedForward(object):

    #Initializes a FeedForward network object based on a genome
    def __init__(self, genome):
        self.genome = genome
        self.config = genome.config
        self.activation = {}
        self.backPropDelta = {}
        self.backPropBiasDelta = {}
        self.backPropMomentum = {}

    #Translates the genome to a FeedForward network, activates the network based on inputs, and returns the output value
    def activate(self, inputs):
        self.activation = {}
        activation = self.activation
        layers = self.genome.layers
        nodes = self.genome.nodes

        for i, layer in sorted(layers.items()):
            if i == 0:
                assert (len(layer) == len(inputs))
                for k, node in enumerate(layer):
                    activation[node] = inputs[k]
            else:
                for node in layer:
                    a = 0
                    for node_in, weight in self.genome.nodes[node].inputs.items():
                        a += weight[0]*activation[node_in]
                    activation[node] = nodes[node].func(a + nodes[node].bias[0])
        return [activation[i+self.config.num_in] for i in range(self.config.num_out)]
    
    def backProp(self, inputs, targets):
        if self.config.bp_miniBatchSize >= len(inputs):
            self.config.bp_miniBatchSize = len(inputs)
            self.batchBackProp(self.config.bp_miniBatchSize, inputs, targets)
        else:
            self.batchBackProp(self.config.bp_miniBatchSize, inputs, targets)
    
    def onlineBackProp(self, target):
        #Define delta for different errorFunc+Activation pairs
        if 'MSE' in self.config.bp_error_func:
            def deltaFunc(t,o):
                if self.config.activation_func == 'sigmoid':
                    return (t-o) * (o * (1-o))

        elif 'CrossEntropy' in self.config.bp_error_func:
            def deltaFunc(t,o):
                if self.config.activation_func == 'sigmoid':
                    return (t-o)
        
        #Initialize variables
        self.backPropDelta = {}
        self.backPropBiasDelta = {}
        activation = self.activation
        layers = self.genome.layers
        nodes = self.genome.nodes
        backPropDelta = self.backPropDelta
        layerKeys = sorted(list(layers.keys()), reverse=True)
        prevKey = None

        #Loop through layers
        for key in layerKeys:
            layer = layers[key]

            #If Output layer
            if key == 2:
                assert (len(layer) == self.config.num_out)
                #Loop through nodes in output layer
                for j,node in enumerate(layer):
                    backPropDelta[node] = deltaFunc(target[j], activation[node])
            
            #If not output layer
            else:
                prevLayer = layers[prevKey]
                #Loop through nodes in layer
                for j, node in enumerate(layer):
                    #If input node
                    if nodes[node].inputs == {}:
                        continue
                    #I not input layer
                    else:
                        delta = 0.0
                        #Loop through nodes in previous layer (reversed so actually the successor layer)
                        for l, nodePrev in enumerate(prevLayer):
                            #If current node have connection to previous layer (reversed so actually the successor layer)
                            if node in list(nodes[nodePrev].inputs.keys()):
                                delta += backPropDelta[nodePrev] * nodes[nodePrev].inputs.get(node)[0]
                            #If not a connection do nothing
                            else:
                                pass

                            #Calculate delta for that node
                            backPropDelta[node] = delta * nodes[node].func(activation[node], gradient = True)
                    
            #Assign just completed key to be the previous key and then continue
            prevKey = key

    def batchBackProp(self, batchSize, inputs, targets):
        inputs, targets = copy.deepcopy(inputs), copy.deepcopy(targets)
        if 'MSE' in self.config.bp_error_func:
            def errorFunc(t,o):
                if self.config.activation_func == 'sigmoid':
                    return (t-o) * (o * (1-o))

        elif 'CrossEntropy' in self.config.bp_error_func:
            def errorFunc(t,o):
                if self.config.activation_func == 'sigmoid':
                    return (t-o)
        
        batchRatio = int(len(inputs)/batchSize)
        batchSize = int(len(inputs)/batchRatio)

        self.backPropDeltaBatch = []
        self.activationBatch = []

        if self.config.bp_miniBatchSize >= len(inputs):
            pass
        else:
            zipped = list(zip(inputs,targets))
            random.shuffle(zipped)
            inputs[:], targets[:] = zip(*zipped)
        
        for batch in range(batchRatio):
            inBatch, tarBatch = inputs[batch*batchSize:(batch+1)*batchSize], targets[batch*batchSize:(batch+1)*batchSize]
            for i, t in zip(inBatch, tarBatch):

                self.backPropDelta = {}
                self.activate(i)
                self.activationBatch.append(copy.deepcopy(self.activation))

                self.onlineBackProp(t)

                self.backPropDeltaBatch.append(copy.deepcopy(self.backPropDelta))

            if 'gradientDescent' in self.config.bp_algorithm:
                self.gradientDesc(True)
            else:
                raise NotImplementedError("Only 'gradientDescent' bp algorithm is implemented as of now.")
            
    def gradientDesc(self, miniBatch = False):
        if miniBatch == True:
            nodes = self.genome.nodes
            layers = self.genome.layers
            lr = self.config.bp_lr
            mu = self.config.bp_momentum
            for i, layer in layers.items():
                if i == 0:
                    pass
                else:
                    for node in layer:
                        for node_in, weight in nodes[node].inputs.items():
                            errorSum = 0
                            for activation, backPropDelta in zip(self.activationBatch, self.backPropDeltaBatch):
                                errorSum += activation[node_in] * backPropDelta[node]
                            weightDelta = lr * errorSum/len(self.activationBatch) + weight[1] * mu
                            weight[0] += weightDelta
                            weight[1] = weightDelta
                        biasErrorSum  = 0
                        for backPropDelta in self.backPropDeltaBatch:
                            biasErrorSum += backPropDelta[node]
                        biasDelta = lr * biasErrorSum/len(self.activationBatch) + nodes[node].bias[1] * mu
                        nodes[node].bias[0] += biasDelta
                        nodes[node].bias[1] = biasDelta

        else:
            backPropDelta = self.backPropDelta
            activation = self.activation
            nodes = self.genome.nodes
            layers = self.genome.layers
            lr = self.config.bp_lr
            mu = self.config.bp_momentum
            for i, layer in layers.items():
                if i == 0:
                    pass
                else:
                    for node in layer:
                        for node_in, weight in nodes[node].inputs.items():
                            weightDelta = lr * activation[node_in] * backPropDelta[node] + weight[1] * mu
                            weight[0] += weightDelta
                            weight[1] = weightDelta
                        biasDelta = lr * backPropDelta[node] + nodes[node].bias[1] * mu
                        nodes[node].bias[0] += biasDelta
                        nodes[node].bias[1] = biasDelta
