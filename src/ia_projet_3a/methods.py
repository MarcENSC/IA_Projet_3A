import subprocess
import socket
import json

def press(action):
    subprocess.Popen(['xdotool', 'keydown', action.value])

def unpress(action):
    subprocess.Popen(['xdotool', 'keyup', action.value])

def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    return client_socket

def extract_floors_and_stocks(filename):
    # load json
    with open(filename, 'r') as file:
        data = json.load(file)

    floors = []
    stocks = []

    # Parcourir les aires et les créations pour récupérer les données
    for area in data['areas']:
        for creation in area['creation']:
            # Vérifier si l'élément est un "Floor"
            if 'macro' in creation and creation['macro'] == 'Floor':
                floors.append(creation)
            
            # Vérifier si l'élément est un "Block" ou "Brick" avec un contenu
            if 'thing' in creation and creation['thing'] in ['Block', 'Brick']:
                if 'contents' in creation:
                    stocks.append(creation)

    return floors, stocks


def is_above_floor(posX, floors):
    for floor in floors:
        x = floor.get('x', 0)
        width = floor['width']
        if x - 8 <= posX < x + width:
            return True
    return False
