from mario_ai.utils import nn_save_manager
from utils import logger
from game_manager import simulation
from ai import neural_network
from ai.individual import Individual
import random as rnd
import torch

def train(nb_ind, best_ind_ratio, mutation_rate, mutation_range):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger.log("Starting the game AI...")
    
    nb_best_ind = int(nb_ind * best_ind_ratio)
    nb_gen = 1

    # ind = Individual(neural_network.NN(), 0)
    population = [Individual(neural_network.NN(), 0) for _ in range(nb_ind)]

    while True:
        # Evaluate
        actual_ind = 1
        for ind in population:
            simulation.start_simulation(ind)
            print(f"{actual_ind}/{nb_ind} - ID {ind.id} - Score {ind.score}")
            actual_ind += 1
        
        # Select Best
        population.sort(key=lambda x: -x.score)
        # best_individuals = []
        # for _ in range(nb_best_ind):
        #     ind1, ind2 = rnd.sample(population, 2)
        #     best_ind = ind1 if ind1.score > ind2.score else ind2
        #     best_individuals.append(best_ind)
        #     population.remove(best_ind)

        best_individuals = population[:nb_best_ind]
        best_score = best_individuals[0].score
        nn_save_manager.export_nn_to_json(best_individuals[0].neural_network, "saves/nn.json")

        # Reproduce
        children = []
        for _ in range(nb_ind - nb_best_ind):
            ind1, ind2 = rnd.sample(best_individuals, 2)

            child = Individual(neural_network.NN(), 0)
            child.cross(ind1, ind2)
            child.mutate(mutation_rate, mutation_range)

            children.append(child)

        population = best_individuals + children

        logger.log(f"Generation {nb_gen} finished ! Best score : {best_score}")
        nb_gen += 1