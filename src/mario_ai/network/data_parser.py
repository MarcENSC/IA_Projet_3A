def parse_game_data(data,map_mat):
    """ Parser les données reçues du serveur (position du joueur, des ennemis, etc.) """
    data_parts = data.decode('utf-8').strip().split(',')

    data = {
        'player_position_x': 0,
        'player_position_y': 0,
        'ecarts' : [0 for _ in range(5)],
        'ecart_player_enemy_1': 0,
        'map_state': [0 for _ in range(80)],
        'player_x_speed': 0,
        'player_y_speed': 0,
        'nb_enemies': 0
    }
    
    try:
        player_x_pos, player_y_pos = map(int, data_parts[:2])
        player_page = int(data_parts[2])
        player_X_speed = int(data_parts[3])
        player_Y_speed = int(data_parts[4])
        e1,e2,e3,e4,e5 = map(int, data_parts [5:10])
        f1,f2,f3,f4,f5 = map(int, data_parts [10:15])
        ep1,ep2,ep3,ep4,ep5 = map(int,data_parts[15:20])

        enemy_flags = [f1,f2,f3,f4,f5]
        enemies_positions_x = [
            256 * ep1 + e1,
            256 * ep2 + e2,
            256 * ep3 + e3,
            256 * ep4 + e4,
            256 * ep5 + e5
        ]
        player_x = 256 * player_page + player_x_pos
        player_y = player_y_pos

        ecarts = [(enemies_positions_x[i]-player_x)/200 if enemy_flags[i] else 1 for i in range(5)]

        nb_enemies = (f1 + f2 + f3 + f4 + f5) / 5

        player_x_bloc = int((player_x + 8) / 16)
        player_y_bloc = 16 - int((player_y + 8)/ 16) - 3
        
        view = 4

        map_state = []
        for i in range(-view+1, view + 2):
            # print("")
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
                        
        data['player_position_x'] = player_x / 3150
        data['player_position_y'] = player_y / 180
        data['ecarts'] = ecarts
        data['map_state'] = map_state
        data['player_x_speed'] = player_X_speed
        data['player_y_speed'] = player_Y_speed
        data['nb_enemies'] = nb_enemies

    except Exception as e:
        0
        # print(f"Error: {e}")

    # Retourner un état de jeu structuré
    return data
