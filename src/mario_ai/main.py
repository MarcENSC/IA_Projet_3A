import time
import subprocess
import torch
from game_manager import controls, environment, map_manager
from game_manager.actions import Action
from ai import neural_network
from network import server, data_parser
from utils import logger

def main():
    logger.clear()
    logger.log("Starting the game IA...")

    # Lancer le jeu (C++ côté serveur)
    logger.log("Launching game...")
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
    time.sleep(2)

    # Connexion au serveur
    client = server.connect_to_server()
    logger.log("Connected to game server")

    # Charger la carte
    filename = 'maps/World11.json'
    map_matrix = map_manager.parse_json_to_matrix(filename)

    # Initialiser l'agent IA
    NN = neural_network.NN()

    # Initialiser l'output
    action_dict = {
        'up' : {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_UP},
        'down': {'need_press': False, 'is_pressed': False, 'action_type': Action.CROUCH},
        'left': {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_LEFT},
        'right': {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_RIGHT},
        'run': {'need_press': False, 'is_pressed': False, 'action_type': Action.RUN},
        'jump': {'need_press': False, 'is_pressed': False, 'action_type': Action.JUMP},
    }
    logger.log("Dictionnary initialized")

    time.sleep(2)
    controls.press(Action.ENTER)
    time.sleep(3)

    logger.log("Looping now")

    t=0

    while t<500:
        # Recevoir les données du serveur
        data = client.recv(4096)
        if not data:
            break

        game_state = data_parser.parse_game_data(data,map_matrix)
        logger.log(game_state)
        
        input = [game_state['player_position_x'],
                 game_state['player_position_y'],
                 game_state['ecart_player_enemy_1']] + game_state['map_state']
        
        input_tensor = torch.tensor(input, dtype=torch.float32)

        # Décider de l'action à prendre
        action = NN.forward(input_tensor).tolist()

        action_dict['up']['need_press'] = action[0]
        action_dict['down']['need_press'] = action[1]
        action_dict['left']['need_press'] = action[2]
        action_dict['right']['need_press'] = action[3]
        action_dict['run']['need_press'] = action[4]
        action_dict['jump']['need_press'] = action[5]

        # Exécuter l'action (simuler la pression de la touche)
        action_dict = controls.handle_action_request(action_dict)

        # logger.log(action_dict)
        t+=1
    
    controls.stop()
    client.close()
    logger.log("Game Over")
    main()

if __name__ == "__main__":
    main()
