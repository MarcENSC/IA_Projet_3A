from ai.neural_network import cross
import random as rnd
import torch

class Individual:
    def __init__(self, neural_network, score=0.0):
        self.neural_network = neural_network
        self.score = score

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_neural_network(self):
        return self.neural_network

    def cross(self,ind1,ind2):
        self.neural_network = cross(ind1.neural_network,ind2.neural_network, rnd.random())

    def mutate(self, m_rate, m_range):
        params = list(self.neural_network.parameters())
        for p in params:
            if rnd.random() < m_rate:
                mutation_factor = torch.normal(mean=torch.zeros_like(p), std=m_range * torch.ones_like(p))
                p.data += mutation_factor
                p.data = torch.clamp(p.data, min=-1, max=1)