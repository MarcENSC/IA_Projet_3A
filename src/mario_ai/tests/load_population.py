import sys
sys.path.append("..")

from utils.nn_save_manager import *

def main():
    population = load_population("../saves/genetic_algorithm/training1737128432/389")
    print(population)


if __name__ == "__main__":
    main()