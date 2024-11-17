def is_above_floor(posX, floors):
    """ Vérifie si le joueur est au-dessus du sol """
    for floor in floors:
        x = floor.get('x', 0)
        width = floor['width']
        if x - 8 <= posX < x + width:
            return True
    return False
