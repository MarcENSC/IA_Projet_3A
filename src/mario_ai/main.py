from ai import genetic_algorithm, neat
from game_manager import simulation
from utils import logger

def main():
    # simulation.load_best_simulation()

    nb_ind = 100
    best_ind_ratio = 0.1
    mutation_rate = 0.01
    mutation_range = 0.1
    nn_format = [144+8, 6]
    pretrained_pop = ""
    # pretrained_pop = "genetic_algorithm/training1736846704/204"
    genetic_algorithm.train(nb_ind, best_ind_ratio, mutation_rate, mutation_range, nn_format, pretrained_pop)


if __name__ == "__main__":
    logger.clear()
    main()
