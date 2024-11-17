import json

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
