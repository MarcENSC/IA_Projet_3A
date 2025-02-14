from utils import logger, nn_save_manager
from game_manager import simulation
from ai import sequential_neural_network, recurrent_neural_network
from ai.individual import Individual
import random as rnd

def initialize_population(nb_ind, nn_format, pretrained_pop):
    if pretrained_pop:
        pop = nn_save_manager.load_population(f"saves/{pretrained_pop}")
        pop += reproduce(pop, nb_ind, len(pop), nn_format, 0.001, 0.001)
    else:
        pop = [Individual(sequential_neural_network.NN(nn_format), 0) for _ in range(nb_ind)]
    
    return pop

    # ind = Individual(sequential_neural_network.NN(nn_format), 0)
    # return [ind for _ in range(nb_ind)]

    # return [Individual(recurrent_neural_network.RNN(nn_format), 0) for _ in range(nb_ind)]

def evaluate_population(children):
    actual_ind = 1
    for ind in children:
        simulation.start_simulation(ind)
        print(f"{actual_ind}/{len(children)} - ID {ind.id} - Score {ind.score}%")
        actual_ind += 1

def select_best_individuals(population, nb_ind):
    population.sort(key=lambda x: -x.score)
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
    return new_population_best + new_population_worst

def reproduce(best_individuals, nb_ind, nb_best_ind, nn_format, mutation_rate, mutation_range):
    # get individuals NN instance (sequential or recurrent)
    type = best_individuals[0].nn_type
    children = []
    for _ in range(nb_ind - nb_best_ind):
        ind1, ind2 = rnd.sample(best_individuals, 2)

        match type:
            case "NN":
                child = Individual(sequential_neural_network.NN(nn_format), 0)
            case "RNN":
                child = Individual(recurrent_neural_network.RNN(nn_format), 0)
            case _:
                raise ValueError("Error: unknown neural network type, giving parent1_nn to child")

        child.cross(ind1.neural_network, ind2.neural_network, rnd.random())
        child.mutate(mutation_rate, mutation_range)
        children.append(child)
    return children

def neat_reproduce(best_individuals, nb_ind, nb_best_ind, nn_format, mutation_rate, neuron_mutation_rate, layer_mutation_rate, mutation_range):
    children = []
    for _ in range(nb_ind - nb_best_ind):
        ind1, ind2 = rnd.sample(best_individuals, 2)
        child = Individual(sequential_neural_network.NN(nn_format), 0)
        child.neat_cross(ind1.neural_network, ind2.neural_network, rnd.random())
        child.mutate_neat(mutation_rate, neuron_mutation_rate, layer_mutation_rate, mutation_range)
        children.append(child)
    return children

def save_best_individuals(best_individuals, training_type, training_id, nb_gen):
    for p in best_individuals:
        nn_save_manager.export_nn_to_json(p, training_type, training_id, nb_gen, f"{p.id}_{int(p.score)}.json")

def log_generation(nb_gen, best_score, best_id, moy):
    print(2 * f"{'=' * 25}\n", end="")
    logger.log(f"\nGeneration : {nb_gen} finished !\nBest score : {best_score}\nBest ID : {best_id}\nMean best scores : {moy}")
    print(2 * f"{'=' * 25}\n")