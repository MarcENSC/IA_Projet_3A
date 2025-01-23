from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, nn_format):
        super().__init__()
        self.nn_format = nn_format
        
        self.input_size = nn_format[0]
        self.output_size = nn_format[-1]
        
        self.layers = [[None for _ in range(len(nn_format)-1)] for _ in range(len(nn_format)-1)]

        for i in range(len(nn_format)-1):
            for j in range(len(nn_format)-(i+1)):
                self.layers[i][j] = nn.Linear(nn_format[i], nn_format[j+i+1])
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        n = len(self.nn_format)
        layers_output = [0 for _ in range(n)]

        layers_output[0] += input

        for i in range(n-1):
            for j in range(n-(i+1)):
                layers_output[i+j+1] += self.layers[i][j](layers_output[i])

        output = layers_output[-1]
        output = self.sigmoid(output)
        bool_outputs = output > 0.5

        return bool_outputs
    
    def mutate(self, m_rate, m_range):
        pass

def cross(parent1_nn, parent2_nn, alpha=0.5):
    return parent1_nn