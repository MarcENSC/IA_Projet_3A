from torch import nn
import random as rnd
import copy

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(48 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probabilities = self.sigmoid(logits)
        bool_outputs = probabilities > 0.5
        return bool_outputs

def cross(parent1_nn, parent2_nn, alpha=0.5):
    # Clone the parent networks so that they are independent copies
    parent1_nn = copy.deepcopy(parent1_nn)
    parent2_nn = copy.deepcopy(parent2_nn)

    # Get the parameters (weights and biases) from both parent neural networks
    parent1_params = list(parent1_nn.parameters())
    parent2_params = list(parent2_nn.parameters())
    
    # Create a list to hold the new child parameters
    child_params = []
    
    # Perform linear combination for each parameter (layer's weights/biases)
    for param1, param2 in zip(parent1_params, parent2_params):
        # Compute the new parameter using alpha and (1-alpha)
        child_param = alpha * param1 + (1 - alpha) * param2
        child_params.append(child_param)
    
    # Create a new NN (child) and assign the new parameters
    child_nn = NN()  # This assumes NN is defined as in your original code
    
    # Load the new parameters into the child's neural network
    child_nn_params = child_nn.parameters()
    
    # The parameters are stored as a list, so we need to zip them together
    for child_param, new_param in zip(child_nn_params, child_params):
        child_param.data = new_param.data
    
    return child_nn