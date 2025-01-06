from torch import nn
import torch
import random as rnd

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
    
    def shape(self):
        return [param.size() for param in self.parameters()]
    
    def add_neuron(self, layer_m_rate):
        if rnd.random() < layer_m_rate or len(self.nn_format) <= 2:
            layer_index = self.add_layer()
        else:
            # Select a random layer
            layer_index = rnd.randint(0, len(self.nn_format) - 3)
        
        layer = self.linear_relu_stack[layer_index * 2] # *2 because of ReLus
        self.nn_format[layer_index] += 1

        # Update current layer
        new_weight = torch.zeros(layer.weight.size(0)+1, layer.weight.size(1))
        new_weight[:-1, :] = layer.weight

        new_bias = torch.zeros(layer.bias.size(0)+1)
        new_bias[:-1] = layer.bias

        layer.weight = nn.Parameter(new_weight)
        layer.bias = nn.Parameter(new_bias)

        layer.in_features = layer.weight.size(1)
        layer.out_features = layer.weight.size(0)


        # Update next layer
        next_layer = self.linear_relu_stack[(layer_index + 1) * 2]
        next_layer.in_features = next_layer.weight.size(1) + 1
        new_next_weight = torch.zeros(next_layer.weight.size(0), next_layer.weight.size(1)+1)
        new_next_weight[:, :-1] = next_layer.weight

        next_layer.weight = nn.Parameter(new_next_weight)

    def add_layer(self):
        # select layer position in nn_format, les entrées et sorties ne doivent pas changer
        layer_index = rnd.randint(0, len(self.nn_format) - 2)
        