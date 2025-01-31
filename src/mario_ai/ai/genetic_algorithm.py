from statistics import mean
from ai.methods import *
from utils.nn_save_manager import *
import torch
import time
import matplotlib.pyplot as plt

def update_plot(ax, line1, line2, data, nb_gen):
    line1.set_xdata(range(1, nb_gen + 1))
    line1.set_ydata(data["mean_pop_score"])
    line2.set_xdata(range(1, nb_gen + 1))
    line2.set_ydata(data["best_score"])
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

def train(nb_ind, best_ind_ratio, param_mutation_rate, mutation_range, nn_format, pretrained_population=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_id = int(time.time())
    training_type = "genetic_algorithm"
    nb_best_ind = int(nb_ind * best_ind_ratio)

    population = initialize_population(nb_ind, nn_format, pretrained_population)
    children = population

    print(len(children))

    data = {"mean_pop_score": [], "best_score": [], "mean_speed": []}
    nb_gen = 1

    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Genetic Algorithm Progress")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Score")
    line1, = ax.plot([], [], label="Mean Population Score", color="blue")
    line2, = ax.plot([], [], label="Best Score", color="green")
    ax.legend()

    try:
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

            data["best_score"].append(best_score)
            data["mean_pop_score"].append(moy)

            # Update the plot
            update_plot(ax, line1, line2, data, nb_gen)

            nb_gen += 1
    except KeyboardInterrupt:
        print("Training interrupted by user. Closing plot...")
        plt.ioff()
        plt.show()
