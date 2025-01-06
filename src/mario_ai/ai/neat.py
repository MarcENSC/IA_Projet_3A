from statistics import mean
from ai.methods import *
import torch
import time

def train(nb_ind, best_ind_ratio, param_mutation_rate, neuron_mutation_rate, layer_mutation_rate, mutation_range, nn_format):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_id = int(time.time())
    training_type = "neat"
    nb_best_ind = int(nb_ind * best_ind_ratio)

    population = initialize_population(nb_ind, nn_format)
    children = population

    nb_gen = 1
    while True:
        evaluate_population(children)
        
        population.sort(key=lambda x: -x.score)
        best_score = population[0].score
        best_id = population[0].id
        moy = mean([p.score for p in population])

        new_population = select_best_individuals(population, nb_ind)
        best_individuals = new_population[:nb_best_ind]

        children = reproduce(best_individuals, nb_ind, nb_best_ind, nn_format, param_mutation_rate, mutation_range)
        population = best_individuals + children

        save_best_individuals(best_individuals, training_type, training_id, nb_gen)
        log_generation(nb_gen, best_score, best_id, moy)

        nb_gen += 1