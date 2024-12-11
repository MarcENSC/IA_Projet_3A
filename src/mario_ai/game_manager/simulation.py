import time
import subprocess
import torch
from game_manager import controls, map_manager
from game_manager.actions import Action
from ai import neural_network
from network import server, data_parser
from utils import logger
from ai import individual

def start_simulation(ind: individual):
    # Lancer le jeu (C++ côté serveur)
    # logger.log("Launching game...")
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    
    time.sleep(2)

    # Connexion au serveur
    client = server.connect_to_server()
    # logger.log("Connected to game server")


    # Charger la carte
    filename = 'maps/World11.json'
    map_matrix = map_manager.parse_json_to_matrix(filename)

    # Initialiser l'agent IA
    NN = ind.get_neural_network()

    # Initialiser l'output
    action_dict = {
        'up' : {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_UP},
        'down': {'need_press': False, 'is_pressed': False, 'action_type': Action.CROUCH},
        'left': {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_LEFT},
        'right': {'need_press': False, 'is_pressed': False, 'action_type': Action.MOVE_RIGHT},
        'run': {'need_press': False, 'is_pressed': False, 'action_type': Action.RUN},
        'jump': {'need_press': False, 'is_pressed': False, 'action_type': Action.JUMP},
    }
    # logger.log("Dictionnary initialized")

    time.sleep(2)
    controls.press(Action.ENTER)
    time.sleep(3)

    # logger.log("Looping now")

    t=0
    is_still = False
    
    while t<5000:
        data = client.recv(4096)
        if not data:
            break

        game_state = data_parser.parse_game_data(data,map_matrix)
        # logger.log(game_state)
        
        if t%100 == 0:
            if (game_state['player_x_speed'] == 0) and is_still:
                break
            is_still = (game_state['player_x_speed'] == 0)

        input = [game_state['player_position_x'],
                game_state['player_position_y'],
                game_state['ecart_player_enemy_1']] + game_state['map_state']
        # logger.log(input)

        input_tensor = torch.tensor(input, dtype=torch.float32)

        output = NN.forward(input_tensor).tolist()

        action_dict['up']['need_press'] = output[0]
        action_dict['down']['need_press'] = output[1]
        action_dict['left']['need_press'] = output[2]
        action_dict['right']['need_press'] = output[3]
        action_dict['run']['need_press'] = output[4]
        action_dict['jump']['need_press'] = output[5]

        # Exécuter l'action (simuler la pression de la touche)
        action_dict = controls.handle_action_request(action_dict)

        # logger.log(action_dict)
        t+=1

        if 1>game_state['player_position_x']>ind.get_score():
            ind.set_score(game_state['player_position_x'])

    controls.stop()
    client.close()
    # logger.log("Game Over")

    return ind
