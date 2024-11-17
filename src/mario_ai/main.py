import time
import subprocess
from game_manager import controls, environment, map_manager
from ai import neural_network
from network import server, data_parser
from utils import logger

def main():
    logger.clear()
    logger.log("Starting the game IA...")

    # Lancer le jeu (C++ côté serveur)
    logger.log("Launching game...")
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
    time.sleep(2)  # Attendre que le jeu se lance

    # Connexion au serveur
    client = server.connect_to_server()
    logger.log("Connected to game server")

    # Charger la carte
    floors, stocks = map_manager.extract_floors_and_stocks('maps/World11.json')

    # Initialiser l'agent IA
    # NN = neural_network.NN(8,32,5)

    while True:
        # Recevoir les données du serveur
        data = client.recv(1024)
        if not data:
            break
        
        game_state = data_parser.parse_game_data(data)
        logger.log(game_state)

        # Décider de l'action à prendre
        # action = NN.decide_action(game_state)
        
        # Exécuter l'action (simuler la pression de la touche)
        
    client.close()
    logger.log("Game Over")

if __name__ == "__main__":
    main()
