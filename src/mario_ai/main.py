from utils import logger
from game_manager import simulation
from ai import individual, neural_network

def main():
    logger.log("Starting the game AI...")
    
    population = [individual.Individual(neural_network.NN(),0) for _ in range(10)]
    scores = []

    for ind in population:
        simulation.start_simulation(ind)
        scores.append(ind.score)

    print(scores)

if __name__ == "__main__":
    logger.clear()
    main()