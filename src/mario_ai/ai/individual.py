from ai import sequential_neural_network, recurrent_neural_network
import random as rnd
import torch
import copy

class Individual:
    id_counter = 1
    nn_type = None

    def __init__(self, neural_network, score=0.0):
        self.neural_network = neural_network
        self.score = score
        self.id = Individual.id_counter
        Individual.id_counter += 1

        if isinstance(neural_network, sequential_neural_network.NN):
            Individual.nn_type = "NN"
        elif isinstance(neural_network, recurrent_neural_network.RNN):
            Individual.nn_type = "RNN"
        else:
            print("Error: unknown neural network type, setting to None")
            Individual.nn_type = None

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_neural_network(self):
        return self.neural_network

    def cross(self, parent1_nn, parent2_nn, alpha=0.5):
        match self.nn_type:
            case "NN":
                self.neural_network = sequential_neural_network.cross(parent1_nn, parent2_nn, alpha)
            case "RNN":
                self.neural_network = recurrent_neural_network.cross(parent1_nn, parent2_nn, alpha)
            case _:
                print("Error: unknown neural network type, giving parent1_nn to child")
                self.neural_network = parent1_nn
                
    def mutate(self, m_rate, m_range):
        self.neural_network.mutate(m_rate, m_range)
        