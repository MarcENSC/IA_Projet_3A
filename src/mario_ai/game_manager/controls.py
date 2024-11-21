import subprocess
from .actions import Action

def handle_action_request(actions):
    for k in actions.keys():
        val = actions[k]
        if val['need_press'] and not val['is_pressed']:
            press(val['action_type'])
            val['is_pressed'] = True
        elif not val['need_press'] and val['is_pressed']:
            unpress(val['action_type'])
            val['is_pressed'] = False
    return actions

def press(action):
    """ Simuler une pression de touche """
    subprocess.Popen(['xdotool', 'keydown', action.value])

def unpress(action):
    """ Simuler le relâchement d'une touche """
    subprocess.Popen(['xdotool', 'keyup', action.value])

def stop():
    for a in Action:
        unpress(a)