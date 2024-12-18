from game_manager import environment

def parse_game_data(data,map_mat):
    """ Parser les données reçues du serveur (position du joueur, des ennemis, etc.) """
    data_parts = data.decode('utf-8').strip().split(',')
    
    player_x_pos, player_y_pos = map(int, data_parts[:2])
    e1,e2,e3,e4,e5 = map(int, data_parts [2:7])
    f1,f2,f3,f4,f5 = map(int, data_parts [7:12])
    player_page = int(data_parts[12])
    enemy_1_page = int(data_parts[13])
    
    enemy_1_x = 256 * enemy_1_page + e1
    player_x = 256 * player_page + player_x_pos
    player_y = player_y_pos

    ecart = enemy_1_x - player_x
    if not f1:
        ecart = 200

    player_x_bloc = int((player_x + 8) / 16)
    player_y_bloc = 16 - int((player_y + 8)/ 16) - 3
    
    view = 3

    map_state = []
    for i in range(-view+1, view + 2):
        print("")
        for j in range(-view, view + 1):
            if i==j==0:
                0
                # print("M", end=" ")
            else:
                y_index = -player_y_bloc + i
                x_index = player_x_bloc + j
                
                if -len(map_mat) <= y_index < 0 and 0 <= x_index < len(map_mat[0]):
                    map_state.append(map_mat[y_index][x_index])
                    # print(map_mat[y_index][x_index], end=" ")
                else:
                    map_state.append(0)
                    # print("0", end=" ")


    # Retourner un état de jeu structuré
    return {
        'player_position_x': player_x/3150,
        'player_position_y': player_y/180,
        'enemy_1_existence': f1,
        'enemy_1_x_position': enemy_1_x,
        'ecart_player_enemy_1': ecart/200,
        'map_state': map_state
    }
