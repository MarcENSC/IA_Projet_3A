from ai import neural_network
import random as rnd
import torch
import copy

class Individual:
    id_counter = 1

    def __init__(self, neural_network, score=0.0):
        self.neural_network = neural_network
        self.score = score
        self.id = Individual.id_counter
        Individual.id_counter += 1

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_neural_network(self):
        return self.neural_network

    def cross(self, parent1_nn, parent2_nn, alpha=0.5):
        parent1_nn = copy.deepcopy(parent1_nn)
        parent2_nn = copy.deepcopy(parent2_nn)

        parent1_params = list(parent1_nn.parameters())
        parent2_params = list(parent2_nn.parameters())
        
        child_params = []
        
        for param1, param2 in zip(parent1_params, parent2_params):
            child_param = alpha * param1 + (1 - alpha) * param2
            child_params.append(child_param)
        
        child_nn = neural_network.NN(parent1_nn.nn_format)
        
        child_nn_params = child_nn.parameters()
        
        for child_param, new_param in zip(child_nn_params, child_params):
            child_param.data = new_param.data
        
        self.neural_network = child_nn

    def mutate(self, m_rate, m_range):
        params = list(self.neural_network.parameters())
        for p in params:
            if rnd.random() < m_rate:
                mutation_factor = torch.normal(mean=torch.zeros_like(p), std=m_range * torch.ones_like(p))
                p.data += mutation_factor

    def neat_cross(self, parent1_nn, parent2_nn, alpha=0.5):
        self.neural_network = copy.deepcopy(parent1_nn)

    def mutate_neat(self, m_rate, neuron_m_rate, layer_m_rate, m_range):
        params = list(self.neural_network.parameters())
        for p in params:
            if rnd.random() < m_rate:
                mutation_factor = torch.normal(mean=torch.zeros_like(p), std=m_range * torch.ones_like(p))
                p.data += mutation_factor
        if rnd.random() < neuron_m_rate:
            self.neural_network.add_neuron(layer_m_rate)
            print(self.neural_network.shape())
        