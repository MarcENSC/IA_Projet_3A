import json
import numpy as np

def parse_json_to_matrix(filename, cell_size=8):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Détermine la taille maximale pour la matrice
    max_x = 0
    max_y = 0

    for area in data['areas']:
        for item in area['creation']:
            x = item.get('x', 0)
            y = item.get('y', 0)

            if 'width' in item:
                x += item['width']
            if 'height' in item:
                y += item['height']

            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Convertit les tailles en indices de matrice
    max_x //= cell_size
    max_y //= cell_size

    # Initialise la matrice
    matrix = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Remplit la matrice avec les objets
    for area in data['areas']:
        for item in area['creation']:
            x = item.get('x', 0) // cell_size
            y = max_y - (item.get('y', 0) // cell_size)  # Inverser Y pour la matrice

            symbol = item.get('thing', item.get('macro', ' '))
            if symbol == "Floor":
                for i in range(item.get('width', cell_size) // cell_size):
                    matrix[y][x + i] = 1
            elif symbol == "Brick":
                matrix[y][x] = 1
            elif symbol == "Block":
                matrix[y][x] = 1
            elif symbol == "Goomba":
                matrix[y][x] = 0
            elif symbol == "Pipe":
                for i in range(1+item.get('height', cell_size) // cell_size):
                    matrix[y - i][x] = 1
                    matrix[y - i][x+1] = 1
            elif symbol == "Stone":
                for i in range(item.get('height', cell_size) // cell_size):
                    matrix[y - i][x] = 1
            else:
                matrix[y][x] = 0

    return matrix

def extract_floors_and_stocks(filename):
    """ Charger la carte et extraire les informations sur les sols et les objets """
    with open(filename, 'r') as file:
        data = json.load(file)

    floors = []
    stocks = []

    for area in data['areas']:
        for creation in area['creation']:
            if 'macro' in creation and creation['macro'] == 'Floor':
                floors.append(creation)
            if 'thing' in creation and creation['thing'] in ['Block', 'Brick']:
                if 'contents' in creation:
                    stocks.append(creation)

    return floors, stocks
