from ai import genetic_algorithm, neat
from game_manager import simulation
from utils import logger

def main():
    # simulation.load_best_simulation()

    nb_ind = 100
    best_ind_ratio = 0.2
    mutation_rate = 0.1
    mutation_range = 0.5
    nn_format = [144+8, 256, 128, 6]
    # genetic_algorithm.train(nb_ind, best_ind_ratio, mutation_rate, mutation_range, nn_format)

    nb_ind = 4
    best_ind_ratio = 0.5
    mutation_rate = 0.1
    mutation_range = 0.5
    nn_format = [144+8, 256, 128, 6]
    neuron_mutation_rate = 1
    layer_mutation_rate = 0
    neat.train(nb_ind, 
               best_ind_ratio, 
               mutation_rate, 
               neuron_mutation_rate, 
               layer_mutation_rate, 
               mutation_range, 
               nn_format)


if __name__ == "__main__":
    logger.clear()
    main()
