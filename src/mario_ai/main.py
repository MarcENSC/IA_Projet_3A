from ai import genetic_algorithm
from game_manager import simulation
from utils import logger

def main():
    # simulation.load_best_simulation()

    nb_ind = 100
    best_ind_ratio = 0.25  # Between 0 and 1
    mutation_rate = 0.1
    mutation_range = 1
    genetic_algorithm.train(nb_ind, best_ind_ratio, mutation_rate, mutation_range)


if __name__ == "__main__":
    logger.clear()
    main()
