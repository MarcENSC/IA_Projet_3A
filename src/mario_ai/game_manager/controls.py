import subprocess

def press(action):
    """ Simuler une pression de touche """
    subprocess.Popen(['xdotool', 'keydown', action.value])

def unpress(action):
    """ Simuler le relâchement d'une touche """
    subprocess.Popen(['xdotool', 'keyup', action.value])
