from torch import nn

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