from genome import *
import tqdm
import random
from itertools import count
import functools
from multiprocessing import Pool, current_process
import numpy as np

#Global counter to keep track of historical markings
class InnovNumber(object):
    def __init__(self, config):
        self.counter = count(config.num_in + config.num_hidden + config.num_out)
    #Increment counter and returns value
    def next(self):
        return next(self.counter)

class Population(object):

    def __init__(self, config, inputs, targets, evalFunc = None):
        self.config = config
        self.innovNumber = InnovNumber(self.config)
        self.pop = [Genome(config, self.innovNumber) for dummy in range(config.pop_size)]
        for genome in self.pop:
            genome.configNewGenome()
        self.evalFunc = evalFunc
        self.generation = 0
        self.inputs = inputs
        self.targets = targets
        self.evaluations = 0
        self.evalFitness(self.pop)

    def evalFitness(self, pop):
        if isinstance(pop, (np.ndarray, list)):
            for genome in pop:
                self.evalFunc(genome, self.inputs, self.targets)  
        else:
            genome = pop
            self.evalFunc(genome, self.inputs, self.targets)

    def runCHEAT(self):
        self.bestFitness = []
        self.populationFitness = []
        self.avgSize = []

        if 'min' in self.config.fitness_goal:
            self.bestFitness.append(sorted([(g.fitness, g) for g in self.pop], key = lambda x: x[0])[0])
        elif 'max' in self.config.fitness_goal:
            self.bestFitness.append(sorted([(g.fitness, g) for g in self.pop], key = lambda x: x[0], reverse = True)[0])

        self.populationFitness.append(sorted([g.fitness for g in self.pop]))

        self.avgSize.append(np.average([len(g.nodes) for g in self.pop]))

        print('----------Initializaiton----------')
        print('--- Best Fitness: ', self.bestFitness[self.generation][0], '- size: {} nodes'.format(len(self.bestFitness[self.generation][1].nodes)))
        print('--- Average Fitness: ', np.average(self.populationFitness[self.generation]))
        print('')

        if 'min' in self.config.fitness_goal:
            while self.generation < self.config.maxGenerations and self.bestFitness[self.generation][0] > self.config.fitnessThreshold:

                print('----------Generation {}----------'.format(self.generation))

                self.evolutionPhase()
                self.generation += 1

                self.bestFitness.append(sorted([(g.fitness, g) for g in self.pop], key = lambda x: x[0])[0])
                self.populationFitness.append(sorted([g.fitness for g in self.pop]))

                self.avgSize.append(np.average([len(g.nodes) for g in self.pop]))

                print('--- Evaluations: ', self.evaluations)
                print('--- Best Fitness: ', self.bestFitness[self.generation][0], '- size: {} nodes'.format(len(self.bestFitness[self.generation][1].nodes)))
                print('--- Average Fitness: ', np.average(self.populationFitness[self.generation]))
                print('')

        elif 'max' in self.config.fitness_goal:
            while self.generation < self.config.maxGenerations and self.bestFitness[self.generation][0] < self.config.fitnessThreshold:

                print('----------Generation {}----------'.format(self.generation))

                self.evolutionPhase()
                self.generation += 1

                self.bestFitness.append(sorted([(g.fitness, g) for g in self.pop], key = lambda x: x[0], reverse = True)[0])
                self.populationFitness.append(sorted([g.fitness for g in self.pop]))

                self.avgSize.append(np.average([len(g.nodes) for g in self.pop]))

                print('--- Evaluations: ', self.evaluations)
                print('--- Best Fitness: ', self.bestFitness[self.generation][0], '- size: {} nodes'.format(len(self.bestFitness[self.generation][1].nodes)))
                print('--- Average Fitness: ', np.average(self.populationFitness[self.generation]))
                print('')


    
    def evolutionPhase(self):
        self.pop = copy.deepcopy(self.runGA(self.pop))
        self.evaluations += self.config.pop_size
    
    def learningPhase(self, pop):
        # subPops = [[copy.deepcopy(genome) for dummy in range(self.config.sub_pop_size)] for genome in pop]

        #Defines which learning algorithm to run
        if 'BackProp' in self.config.learningAlgorithm:

            subPop = copy.deepcopy(pop)
            # if self.config.parallelized == True:
            #     p = Pool()
            #     subPop,evaluations = zip(*p.map(self.runBackProp, subPop))
            #     p.close()
            #     p.join()
            #     print(sum(evaluations))
            #     self.evaluations += sum(evaluations)
            # else:
            #     subPop,evaluations = self.runBackProp(subPop)
            #     self.evaluations += evaluations

            # trainedPop = copy.deepcopy(subPop)
            ### FOR PLOTTING
            if self.config.parallelized == True:
                p = Pool()
                subPop,evaluations = zip(*p.map(self.runBackProp, subPop))
                p.close()
                p.join()
                print(sum(evaluations))
                self.evaluations += sum(evaluations)
                # import pickle
                # import os
                # import csv
                # f = open(os.path.join('Plots','StoppingAnalysis','{}.csv').format(self.config.num_hidden), 'w', newline='')
                # wr = csv.writer(f)
                # wr.writerow(range(self.config.learning_epochs))
                # for error in errorPopulation:
                #     wr.writerow(error)
                # f.close()
            else:
                subPop,evaluations = self.runBackProp(subPop)
                print(evaluations)
                self.evaluations += evaluations

            trainedPop = copy.deepcopy(subPop)
        
        elif 'GA' in self.config.learningAlgorithm:
            subPops = [[copy.deepcopy(genome) for dummy in range(self.config.sub_pop_size)] for genome in pop]

            if self.config.parallelized == True:
                p = Pool()
                subPops = p.map(functools.partial(self.runParallelizedGA, phase='trainingPhase'), subPops)
                p.close()
                p.join()

            else:
                if self.config.adaptiveLearning == True:
                    learningEpochs = int(self.config.adaptiveLearningParam*self.config.learning_epochs*self.avgSize[-1])
                else:
                    learningEpochs = int(self.config.learning_epochs)

                for spIdx, subPop in enumerate(subPops):
                    epochBar = tqdm.tqdm(desc= 'SubPop {}'.format(spIdx), total=learningEpochs, leave=False)
                    for epoch in range(learningEpochs):
                        subPop = self.runGA(subPop, 'trainingPhase')
                        subPops[spIdx] = subPop
                        epochBar.update()
                    epochBar.close()
            self.evaluations += self.config.pop_size * self.config.sub_pop_size * learningEpochs

            if 'min' in self.config.fitness_goal:
                trainedSubPops = [sorted([(float(g.fitness), g) for g in subPop], key = lambda x: x[0]) for subPop in subPops]
                trainedPop = [trainedSubPop[0][1] for trainedSubPop in trainedSubPops]

            elif 'max' in self.config.fitness_goal:
                trainedSubPops = [sorted([(float(g.fitness), g) for g in subPop], key = lambda x: x[0], reverse = True) for subPop in subPops]
                trainedPop = [trainedSubPop[0][1] for trainedSubPop in trainedSubPops]

        else:
            raise NameError('Chose a valid learning algorithm. Implemented algorithms: [BackProp, GA]')

        return trainedPop

    def runBackProp(self, pop):
        learningEpochs = self.config.learning_epochs

        if self.config.parallelized == True:
            genome = pop
            epoch = 0
            errorWindow1 = []
            errorWindow2 = []
            a = int(self.config.stoppingCritieriaWindowLength/2)
            b = int(self.config.stoppingCritieriaWindowLength/2)
            c = self.config.stoppingCriteria
            while epoch < self.config.learning_epochs:

                genome.network.backProp(self.inputs,self.targets)
                self.evalFitness(genome)

                if self.config.adaptiveLearning == True:
                    if len(errorWindow1) < a:
                        errorWindow1.append(genome.fitness)
                    elif len(errorWindow2) < b:
                        errorWindow2.append(genome.fitness)
                    
                    else:
                        del errorWindow1[0]
                        errorWindow1.append(errorWindow2.pop(0))
                        errorWindow2.append(genome.fitness)

                        if abs(np.mean(errorWindow1) - np.mean(errorWindow2)) < c*(np.mean(errorWindow1) + np.mean(errorWindow2))/2:
                            epoch += 1
                            break
                
                epoch += 1
            return genome, epoch

        else:
            evaluations = 0
            epochBar = tqdm.tqdm(desc = 'Epoch/MaxEpochs', total = learningEpochs*len(pop), leave=False)
            for popIdx, genome in enumerate(pop):
                epoch = 0
                errorWindow1 = []
                errorWindow2 = []
                a = int(self.config.stoppingCritieriaWindowLength/2)
                b = int(self.config.stoppingCritieriaWindowLength/2)
                c = self.config.stoppingCriteria
                while epoch < self.config.learning_epochs:

                    genome.network.backProp(self.inputs,self.targets)
                    self.evalFitness(genome)
                    epochBar.update(1)

                    if self.config.adaptiveLearning == True:

                        if len(errorWindow1) < a:
                            errorWindow1.append(genome.fitness)
                        elif len(errorWindow2) < b:
                            errorWindow2.append(genome.fitness)
                        
                        else:
                            del errorWindow1[0]
                            errorWindow1.append(errorWindow2.pop(0))
                            errorWindow2.append(genome.fitness)

                            if abs(np.mean(errorWindow1) - np.mean(errorWindow2)) < c*(np.mean(errorWindow1) + np.mean(errorWindow2))/2:
                                epoch += 1
                                break
                    
                    epoch += 1
                evaluations += epoch
            return pop, evaluations

        #############FUNCTIONING
        # else:
        #     import matplotlib.pyplot as plt
        #     evaluations = 0
        #     epochBar = tqdm.tqdm(desc = 'Epoch/MaxEpochs', total = learningEpochs*len(pop), leave=False)
        #     for popIdx, genome in enumerate(pop):
        #         epoch = 0
        #         fitnessDiffEpoch1 = None
        #         fitnessPatienceList = []
        #         while epoch < self.config.learning_epochs:
        #             genomeFitnessBefore = genome.fitness
        #             genome.network.backProp(self.inputs,self.targets)
        #             self.evalFitness(genome)
        #             epochBar.update(1)

        #             if self.config.adaptiveLearning == True:
        #                 if fitnessDiffEpoch1 == None:
        #                     fitnessDiffEpoch1 = genome.fitness - genomeFitnessBefore
        #                     fitnessPatienceList.append(fitnessDiffEpoch1)
                        
        #                 if len(fitnessPatienceList) < self.config.stoppingCriteriaPatience:
        #                     fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)
                        
        #                 elif len(fitnessPatienceList) >= self.config.stoppingCriteriaPatience and (sum(fitnessPatienceList)/len(fitnessPatienceList))/fitnessDiffEpoch1 < self.config.stoppingCriteria:
        #                     break

        #                 else:
        #                     del fitnessPatienceList[0]
        #                     fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)

        ###FOR PRINTING AND PLOTTING
        # if self.config.parallelized == True:
        #     genome = pop
        #     epoch = 0
        #     fitnessDiffEpoch1 = None
        #     fitnessPatienceList = []
        #     error = []
        #     while epoch < self.config.learning_epochs:
        #         print(current_process(), 'Epoch:'+str(epoch),'\n')
        #         genomeFitnessBefore = genome.fitness
        #         error.append(genome.fitness)
        #         genome.network.backProp(self.inputs,self.targets)
        #         self.evalFitness(genome)

        #         if self.config.adaptiveLearning == True:
        #             if fitnessDiffEpoch1 == None:
        #                 fitnessDiffEpoch1 = genome.fitness - genomeFitnessBefore
        #                 fitnessPatienceList.append(fitnessDiffEpoch1)
                        
        #             if len(fitnessPatienceList) < self.config.stoppingCriteriaPatience:
        #                 fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)
                    
        #             elif len(fitnessPatienceList) >= self.config.stoppingCriteriaPatience and (sum(fitnessPatienceList)/len(fitnessPatienceList))/fitnessDiffEpoch1 < self.config.stoppingCriteria:
        #                 break

        #             else:
        #                 del fitnessPatienceList[0]
        #                 fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)
                
        #         epoch += 1
        #     return genome, epoch, error

        # else:
        #     import matplotlib.pyplot as plt
        #     evaluations = 0
        #     errorPopulation = []
        #     epochBar = tqdm.tqdm(desc = 'Epoch/MaxEpochs', total = learningEpochs*len(pop), leave=False)
        #     for popIdx, genome in enumerate(pop):
        #         epoch = 0
        #         fitnessDiffEpoch1 = None
        #         fitnessPatienceList = []
        #         error = []
        #         while epoch < self.config.learning_epochs:
        #             genomeFitnessBefore = genome.fitness
        #             error.append(genome.fitness)
        #             genome.network.backProp(self.inputs,self.targets)
        #             self.evalFitness(genome)
        #             epochBar.update(1)
        #             # error.append(genome.fitness)

        #             if self.config.adaptiveLearning == True:
        #                 if fitnessDiffEpoch1 == None:
        #                     fitnessDiffEpoch1 = genome.fitness - genomeFitnessBefore
        #                     fitnessPatienceList.append(fitnessDiffEpoch1)
                        
        #                 if len(fitnessPatienceList) < self.config.stoppingCriteriaPatience:
        #                     fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)
                        
        #                 elif len(fitnessPatienceList) >= self.config.stoppingCriteriaPatience and (sum(fitnessPatienceList)/len(fitnessPatienceList))/fitnessDiffEpoch1 < self.config.stoppingCriteria:
        #                     break

        #                 else:
        #                     del fitnessPatienceList[0]
        #                     fitnessPatienceList.append(genome.fitness- genomeFitnessBefore)
                        
        #                 # if self.config.fitness_goal == 'max':

        #                 #     if fitnessDiffEpoch1 == None:
        #                 #         fitnessDiffEpoch1 = genome.fitness - genomeFitnessBefore
        #                 #         fitnessPatienceList.append(genomeFitnessBefore)

        #                 #     if len(fitnessPatienceList) < self.config.stoppingCriteriaPatience:
        #                 #         fitnessPatienceList.append(genome.fitness)

        #                 #     elif len(fitnessPatienceList) >= self.config.stoppingCriteriaPatience and (sum(fitnessPatienceList)/len(fitnessPatienceList) - genomeFitnessBefore)/fitnessDiffEpoch1 < self.config.stoppingCriteria:
        #                 #         break

        #                 #     else:
        #                 #         del fitnessPatienceList[0]
        #                 #         fitnessPatienceList.append(genome.fitness)

        #                 # elif self.config.fitness_goal == 'min':

        #                 #     if fitnessDiffEpoch1 == None:
        #                 #         fitnessDiffEpoch1 = genomeFitnessBefore - genome.fitness
        #                 #         fitnessPatienceList.append(genomeFitnessBefore)
                            
        #                 #     if len(fitnessPatienceList) < self.config.stoppingCriteriaPatience:
        #                 #         fitnessPatienceList.append(genome.fitness)
        #                 #         print(len(fitnessPatienceList))

        #                 #     elif len(fitnessPatienceList) >=  self.config.stoppingCriteriaPatience and (genomeFitnessBefore - sum(fitnessPatienceList)/len(fitnessPatienceList))/fitnessDiffEpoch1 < self.config.stoppingCriteria:
        #                 #         print(fitnessDiffEpoch1)
        #                 #         print(genomeFitnessBefore - sum(fitnessPatienceList))
        #                 #         break

        #                 #     else:
        #                 #         del fitnessPatienceList[0]
        #                 #         fitnessPatienceList.append(genome.fitness)
        #                 # else:
        #                 #     raise ValueError('fitness_goal needs to be either "min" or "max", not {}'.format(self.config.fitness_goal))
                    
        #             epoch += 1
        #         errorPopulation.append(error)
        #         # plt.plot(range(len(error)), error)
        #         # plt.show()
        #         evaluations += epoch
            
        #     import pickle
        #     import os
        #     import csv
        #     print(errorPopulation[0])
        #     f = open(os.path.join('Plots','StoppingAnalysis','{}.csv').format(self.config.num_hidden), 'w', newline='')
        #     wr = csv.writer(f)
        #     wr.writerow(range(self.config.learning_epochs))
        #     for error in errorPopulation:
        #         wr.writerow(error)
        #     f.close()
            # return pop, evaluations
                    

    def runParallelizedGA(self, pop, phase = 'evolutionPhase'):
        if phase == 'trainingPhase':

            if self.config.adaptiveLearning == True:
                learningEpochs = int(self.config.adaptiveLearningParam*self.config.learning_epochs*self.avgSize[-1])
            else:
                learningEpochs = int(self.config.learning_epochs)

            for epoch in range(learningEpochs):
                pop = self.runGA(pop, phase)
            return pop
        if phase == 'evolutionPhase':
            return self.runGA(pop, phase)

    def runGA(self, pop, phase = 'evolutionPhase'):

        #Define GA parameters depending on which phase
        if phase == 'trainingPhase':
            crossoverRate = self.config.learning_crossover_rate
            popSize = self.config.sub_pop_size
            newRandomGenomeRate = self.config.learning_random_new_rate
        elif phase == 'evolutionPhase':
            crossoverRate = self.config.evolution_crossover_rate
            popSize = self.config.pop_size
            newRandomGenomeRate = self.config.learning_random_new_rate

        #Accquire fitness of genome and sort them from best -> worst
        if 'min' in self.config.fitness_goal:
            popFitness = sorted([(g.fitness, g) for g in pop], key = lambda x: x[0])
        elif 'max' in self.config.fitness_goal:
            popFitness = sorted([(g.fitness, g) for g in pop], key = lambda x: x[0], reverse = True)

        #Define which genomes are suitable for being parents
        if int(len(pop) * crossoverRate) < 2:
            parentGenomes = np.array([gTuple[1] for gTuple in popFitness][:2])
        else:
            parentGenomes = np.array([gTuple[1] for gTuple in popFitness][:int(len(pop) * crossoverRate)])

        #Create new population of blank offspring genomes
        offspringPop = np.array([Genome(self.config, self.innovNumber) for dummy in range(popSize)])

        #Configure the offspring genomes as a crossovergenome between two randomly selected parents from the suitible parent genome pool
        for idx, offspring in enumerate(offspringPop):
            #Configure x nr of the offsprings as completely new random genomes (force population diversity) TODO: Really necessary?
            parents = random.choices(parentGenomes, k = 2)
            offspring.configCrossoverGenome(parents[0], parents[1])
            if idx < int(len(offspringPop) * newRandomGenomeRate):
                offspring.randomizeWeights()

            #Mutate either topology or weights depending on which GA phase
            if phase == 'trainingPhase':
                offspring.mutateWeights()
            elif phase == 'evolutionPhase':
                offspring.mutateTopology()
            else:
                raise NameError('Unexpected phase recieved: {}'.format(phase))

        #Evaluate fitness of the offspring population
        self.evalFitness(offspringPop)

        #If in evolution phase, enter protected childhood habilitation weight training
        if phase == 'evolutionPhase':
            offspringPop = self.learningPhase(offspringPop)

        reproducedPop = np.concatenate((pop, offspringPop))

        #Run duel to kill off genomes until population is halved and return surviving population
        return self.runDuel(reproducedPop, popSize)
    
    def runDuel(self, pop, popSize):
        duellingPop = list(copy.deepcopy(pop))
        survivorPop = []
        while len(survivorPop) < popSize:
            if len(duellingPop) >= 2:
                challenger1 = duellingPop.pop(random.randrange(len(duellingPop)))
                challenger2 = duellingPop.pop(random.randrange(len(duellingPop)))
                fitnessDifference = challenger1.fitness - challenger2.fitness
                if 'min' in self.config.fitness_goal:
                    if fitnessDifference >= 0:
                        survivorPop.append(copy.deepcopy(challenger2))
                    else:
                        survivorPop.append(copy.deepcopy(challenger1))
                elif 'max' in self.config.fitness_goal:
                    if fitnessDifference >= 0:
                        survivorPop.append(copy.deepcopy(challenger1))
                    else:
                        survivorPop.append(copy.deepcopy(challenger2))
            else:
                survivorPop.append(copy.deepcopy(duellingPop[0]))

        return survivorPop