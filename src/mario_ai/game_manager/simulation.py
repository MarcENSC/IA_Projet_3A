import time
import subprocess
import torch
from game_manager import controls, map_manager
from game_manager.actions import Action
from network import server, data_parser
from utils import logger, nn_save_manager
from ai import individual
from statistics import mean

def start_simulation(ind: individual):
    # Lancer le jeu (C++ côté serveur)
    # logger.log("Launching game...")
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    
    time.sleep(1)

    # Connexion au serveur
    client = server.connect_to_server()
    # logger.log("Connected to game server")

    if not client:
        return ind

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

    time.sleep(1)

    # logger.log("Looping now")

    t=0
    velocity_over_time = [0 for _ in range(100)]+[10]
    x_velocity = []

    while t<2500:
        data = client.recv(4096)
        if not data:
            break

        game_state = data_parser.parse_game_data(data,map_matrix)
        # logger.log(game_state)
        

        x_velocity.append(game_state['player_x_speed']*255 / 40)
        speed = abs(255*game_state['player_x_speed']) + abs(255*game_state['player_y_speed'])
        velocity_over_time.append(speed)
        if all(v < 10 for v in velocity_over_time[-100:]):
            break

        input = [game_state['player_x_speed'],
                game_state['player_y_speed'],
                game_state['nb_enemies']] + game_state['map_state'] + game_state['ecarts']

        input_tensor = torch.tensor(input, dtype=torch.float32)

        output = NN.forward(input_tensor).tolist()

        action_dict['up']['need_press'] = output[0]
        action_dict['down']['need_press'] = False
        action_dict['left']['need_press'] = output[2]
        action_dict['right']['need_press'] = output[3]
        action_dict['run']['need_press'] = output[4]
        action_dict['jump']['need_press'] = output[5]

        # Exécuter l'action (simuler la pression de la touche)
        action_dict = controls.handle_action_request(action_dict)
        
        # logger.log(action_dict)
        t+=1
        
        if 1>game_state['player_position_x']>ind.get_score()/100:
            ind.set_score(round(game_state['player_position_x']*100,2))

    print(f"\n== {ind.score} | {mean(x_velocity)} ==")

    # score devient score + score étant donné la moyenne de vitesse le tout normalisé
    ind.score += mean(x_velocity)

    controls.stop()
    client.close()
    # logger.log("Game Over")

    return ind

def load_best_simulation():
    nn = nn_save_manager.load_nn_from_json("saves/nn.json")
    ind = individual.Individual(nn)

    start_simulation(ind)
