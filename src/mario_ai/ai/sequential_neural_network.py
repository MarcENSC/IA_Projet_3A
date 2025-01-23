from torch import nn
import torch
import copy

class NN(nn.Module):
    def __init__(self, nn_format):
        self.nn_format = nn_format
        super().__init__()
        layers = []
        input_size = nn_format[0]
        for output_size in nn_format[1:]:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.pop()  # Remove the last ReLU
        self.linear_relu_stack = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        probabilities = self.sigmoid(logits)
        bool_outputs = probabilities > 0.5
        return bool_outputs
    
    def mutate(self, m_rate, m_range):
        params = list(self.parameters())
        for p in params:
            mutation_mask = torch.rand_like(p) < m_rate
            mutation_factor = torch.normal(mean=0, std=m_range, size=p.size())
            p.data += mutation_mask * mutation_factor

    def shape(self):
        return [param.size() for param in self.parameters()]

def cross(parent1_nn, parent2_nn, alpha=0.5):
    parent1_nn = copy.deepcopy(parent1_nn)
    parent2_nn = copy.deepcopy(parent2_nn)

    parent1_params = list(parent1_nn.parameters())
    parent2_params = list(parent2_nn.parameters())
    
    child_params = []
    
    for param1, param2 in zip(parent1_params, parent2_params):
        child_param = alpha * param1 + (1 - alpha) * param2
        child_params.append(child_param)
    
    child_nn = NN(parent1_nn.nn_format)
    
    child_nn_params = child_nn.parameters()
    
    for child_param, new_param in zip(child_nn_params, child_params):
        child_param.data = new_param.data

    return child_nn