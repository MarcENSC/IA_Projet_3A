from game_manager import environment

def parse_game_data(data):
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
        ecart = -1

    # Retourner un état de jeu structuré
    return {
        'player_position': (player_x, player_y),
        'enemy_1_existence': (f1),
        'enemy_1_x_position': (enemy_1_x),
        'ecart_player_enemy_1': (ecart)
    }
