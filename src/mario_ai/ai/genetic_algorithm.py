from utils import logger,nn_save_manager
from game_manager import simulation
from ai import neural_network
from ai.individual import Individual
from statistics import mean
import random as rnd
import torch
import time

def train(nb_ind, best_ind_ratio, mutation_rate, mutation_range):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_id = int(time.time())
    training_type = "genetic_algorithm"
    
    nb_best_ind = int(nb_ind * best_ind_ratio)

    population = [Individual(neural_network.NN(), 0) for _ in range(nb_ind)]
    children = population

    nb_gen = 1
    while True:
        # Evaluate
        actual_ind = 1
        for ind in children:
            simulation.start_simulation(ind)
            print(f"{actual_ind}/{len(children)} - ID {ind.id} - Score {ind.score}%")
            actual_ind += 1
        
        # Select Best
        population.sort(key=lambda x: -x.score)

        best_score = population[0].score
        best_id = population[0].id
        moy = mean([p.score for p in population])

        new_population_best = []
        new_population_worst = []
        for _ in range(nb_ind // 2):
            ind1, ind2 = rnd.sample(population, 2)
            best_ind, worst_ind = (ind1, ind2) if ind1.score > ind2.score else (ind2, ind1)
            new_population_best.append(best_ind)
            new_population_worst.append(worst_ind)
            population.remove(ind1)
            population.remove(ind2)

        new_population_best.sort(key=lambda x: -x.score)
        new_population_worst.sort(key=lambda x: -x.score)

        new_population = new_population_best + new_population_worst
        best_individuals = new_population[:nb_best_ind]

        # Reproduce
        children = []
        for _ in range(nb_ind - nb_best_ind):
            ind1, ind2 = rnd.sample(best_individuals, 2)

            child = Individual(neural_network.NN(), 0)
            child.cross(ind1, ind2)
            child.mutate(mutation_rate, mutation_range)

            children.append(child)

        population = best_individuals + children
        # print([(p.score,p.id) for p in population])

        for p in best_individuals:
            nn_save_manager.export_nn_to_json(p.neural_network, training_type, training_id, nb_gen, f"{p.id}_{int(p.score)}.json")

        print(2 * f"{'=' * 25}\n",end="")
        logger.log(f"\nGeneration : {nb_gen} finished !\nBest score : {best_score}\nBest ID : {best_id}\nMean score : {moy}\n")
        print(2 * f"{'=' * 25}\n")
        nb_gen += 1