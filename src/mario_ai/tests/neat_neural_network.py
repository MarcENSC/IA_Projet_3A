from torch import nn
import torch

# class StaticNN(nn.Module):
#     def __init__(self, nn_format):
#         super().__init__()
#         self.nn_format = nn_format
        
#         self.input_size = nn_format[0]
#         self.hidden_size_1 = nn_format[1]
#         self.hidden_size_2 = nn_format[2]
#         self.output_size = nn_format[3]
        
#         self.input_to_hidden1 = nn.Linear(self.input_size, self.hidden_size_1)
#         self.input_to_hidden2 = nn.Linear(self.input_size, self.hidden_size_2)
#         self.input_to_output = nn.Linear(self.input_size, self.output_size)

#         self.hidden1_to_hidden2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
#         self.hidden1_to_output = nn.Linear(self.hidden_size_1, self.output_size)

#         self.hidden2_to_output = nn.Linear(self.hidden_size_2, self.output_size)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input):
#         hidden1 = self.relu(self.input_to_hidden1(input))

#         hidden2_from_hidden1 = self.relu(self.hidden1_to_hidden2(hidden1))
#         hidden2_from_input = self.relu(self.input_to_hidden2(input))
#         hidden2 = hidden2_from_hidden1 + hidden2_from_input

#         output_from_input = self.input_to_output(input)
#         output_from_hidden1 = self.hidden1_to_output(hidden1)
#         output_from_hidden2 = self.hidden2_to_output(hidden2)
#         output = output_from_input + output_from_hidden1 + output_from_hidden2

#         final_output = self.sigmoid(output)
#         bool_outputs = final_output > 0.5
#         return bool_outputs

class DynamicNN(nn.Module):
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



nn_format = [152, 8, 6]
input = torch.randn(1, nn_format[0])

nn = DynamicNN(nn_format)
print(nn)
output = nn(torch.randn(1, nn_format[0]))
print("Sortie du réseau :", output)

# nn = StaticNN(nn_format)
# print(nn)
# output = nn(input)
# print("Sortie du réseau :", output)
