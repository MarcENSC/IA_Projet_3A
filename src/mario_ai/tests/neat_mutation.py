from ai.individual import Individual
from ai.neural_network import NN

def main():
    nn_format = [144+8, 256, 128, 6]
    ind = Individual(NN(nn_format))
    ind.mutate_neat(0.1, 1, 0.01, 1)

if __name__ == "__main__":
    main()