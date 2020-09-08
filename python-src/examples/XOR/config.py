class Config(object):
    def __init__(self):

        ###GENERAL SETTINGS###
        self.parallelized = True #Not worth it for BP unless >10000 epochs or >1000 pop_size (Due to overhead when launching child processes)

        ###TOPOLOGY SETTINGS###
        self.num_in = 2
        self.num_out = 1
        self.num_hidden = 0
        self.conn_type = 'full'
        self.fullyConnectedLayers = True

        self.bias_val_sigma = 5
        self.bias_val_mean = 0
        self.bias_val_max = 30
        self.activation_func = 'sigmoid' #Alternatives: sigmoid, tanh

        self.weight_val_sigma = 5
        self.weight_val_mean = 0
        self.weight_val_max = 30

        ###MUTATUION SETTINGS###
        self.adaptiveGrowth = True
        self.adaptiveGrowthParam = 0.01

        self.depthBreadthRatio = 0.01

        self.mutate_weight_prob = 1
        self.mutate_weight_width = 3

        self.mutate_add_node = 0.1
        self.mutate_add_conn = 0.5

        ###POPULATION SETTINGS###
        self.pop_size = 16
        self.sub_pop_size = 1
        self.fitness_goal = 'min'
        self.maxGenerations = 30
        self.fitnessThreshold = 0.03

        ###LEARNINGPHASE SETTINGS####
        self.learningAlgorithm = 'BackProp' #Alternatives: BackProp, GA
        # self.learningAlgorithm = 'GA'
        self.bp_algorithm = 'gradientDescent' #Only needed for BP
        self.bp_lr = 0.3 #Only needed for BP
        self.bp_momentum = 0.1 #Only needed for BP
        self.bp_error_func = 'CrossEntropy' #Alternatives: MSE, CrossEntropy
        self.bp_miniBatchSize = 1

        self.adaptiveLearning = True
        self.stoppingCriteria = 0.000005
        self.stoppingCritieriaWindowLength = 20
        self.learning_epochs = 10000

        self.learning_crossover_rate = 0.2 #Only needed for GA learning
        self.learning_random_new_rate = 0.2 #Only needed for GA learning ##Maybe shouldn't have?

        ###EVOLUTIONPHASE SETTINGS###
        self.evolution_crossover_rate = 0.2
        # self.compatibility_disjoint_factor = 0.1
        
