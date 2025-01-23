import sys

sys.path.append("..")

from ai.individual import Individual
from ai.sequential_neural_network import NN
import copy

def main():
    #make a script that ceates an individual, mutates it and prints the differences between the two neural networks weights
    nn_format = [152, 256, 128, 6]
    ind = Individual(NN(nn_format))
    old_nn = copy.deepcopy(ind.neural_network)
    ind.mutate(0.2, 1)
    new_nn = ind.neural_network

    for old_param, new_param in zip(old_nn.parameters(), new_nn.parameters()):
        print(old_param.data - new_param.data)


if __name__ == "__main__":
    main()