from utils import logger
from game_manager import simulation
from ai import neural_network
from ai.individual import Individual
import random as rnd

def main():
    logger.log("Starting the game AI...")
    
    nb_ind = 10
    best_ind_ratio = 0.2 # between 0 and 1
    nb_best_ind = int(nb_ind * best_ind_ratio)
    mutation_rate = 0.5
    mutation_range = 1
    nb_gen = 1

    population = [Individual(neural_network.NN(), 0) for _ in range(nb_ind)]

    while True:
        # Evaluate
        actual_ind = 1
        for ind in population:
            print(f"{actual_ind}/{nb_ind}")
            actual_ind+=1
            simulation.start_simulation(ind)
        
        # Select Best
        population.sort(key=lambda x: -x.score)
        best_score = population[0].score

        best_individuals = [population[0]]
        population.remove(population[0])
        for _ in range(nb_best_ind-1):
            ind1, ind2 = rnd.sample(population, 2)
            best_ind = ind1 if ind1.score > ind2.score else ind2
            best_individuals.append(best_ind)
            best_ind.score = 0
            population.remove(best_ind)

        # Reproduce
        children = []
        for _ in range(nb_ind - nb_best_ind):
            ind1, ind2 = rnd.sample(best_individuals, 2)

            child = Individual(neural_network.NN(), 0)
            child.cross(ind1, ind2)
            # child.mutate(mutation_rate, mutation_range)

            children.append(child)

        population = best_individuals + children

        logger.log(f"Generation {nb_gen} finished ! Best score : {best_score}")
        nb_gen += 1

        


if __name__ == "__main__":
    logger.clear()
    main()