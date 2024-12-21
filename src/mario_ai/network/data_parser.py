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
        player_X_speed = int(data_parts[3])/255
        player_Y_speed = int(data_parts[4])/255
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
        
        x_view_forward = 10
        x_view_backward = 1
        y_view_up = 4
        y_view_down = 7

        # x_view_forward = 2
        # x_view_backward = 2
        # y_view_up = 2
        # y_view_down = 2

        nb_x = 1 + x_view_forward + x_view_backward
        nb_y = 1 + y_view_up + y_view_down
        total_values = nb_x * nb_y

        # print(total_values)

        map_state = [[0 for _ in range(nb_x)] for _ in range(nb_y)]
        for i in range(-y_view_up,y_view_down+1):
            for j in range(-x_view_backward, x_view_forward+1):
                i_state = i + y_view_up
                j_state =  j + x_view_backward
                if i==j==0:
                    0
                else:
                    y_index = -player_y_bloc + i
                    x_index = player_x_bloc + j
                    
                    if -len(map_mat) <= y_index < 0 and 0 <= x_index < len(map_mat[0]):
                        map_state[i_state][j_state] = map_mat[y_index][x_index]
                    else:
                        if map_state[i_state-1][j_state] == 1:
                            map_state[i_state][j_state] = 1
                        else:
                            map_state[i_state][j_state] = 0
        
        # for i in map_state:
        #     print("")
        #     for j in i:
        #         print(j if  j==1 else ".",end=" ")

        data['player_position_x'] = player_x / 3150
        data['player_position_y'] = player_y / 180
        data['ecarts'] = ecarts
        data['map_state'] = map_state
        data['player_x_speed'] = player_X_speed
        data['player_y_speed'] = player_Y_speed
        data['nb_enemies'] = nb_enemies

    except Exception as e:
        0
        print(f"Error: {e}")

    # Retourner un état de jeu structuré
    return data
