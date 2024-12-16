from utils import logger, save
from game_manager import simulation
from ai import neural_network
from ai.individual import Individual
import random as rnd
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger.log("Starting the game AI...")
    
    nb_ind = 100
    best_ind_ratio = 0.2  # Between 0 and 1
    nb_best_ind = int(nb_ind * best_ind_ratio)
    mutation_rate = 0.2
    mutation_range = 2
    nb_gen = 1

    # ind = Individual(neural_network.NN(), 0)
    population = [Individual(neural_network.NN(), 0) for _ in range(nb_ind)]

    while True:
        # Evaluate
        actual_ind = 1
        for ind in population:
            ind.set_score(0)
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
        save.export_nn_to_json(best_individuals[0].neural_network, "nn.json")

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

if __name__ == "__main__":
    logger.clear()
    main()