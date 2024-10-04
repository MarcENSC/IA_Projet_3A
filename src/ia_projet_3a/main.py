import subprocess
import time
from enum import Enum

class Action(Enum):
    MOVE_RIGHT = 'Right'
    MOVE_LEFT = 'Left'
    CROUCH = 'Down'
    RUN = 'Z'
    JUMP = 'X'
    ENTER = 'Return'
    ESCAPE = 'Escape'

def perform_action(actions, duration):
    """Exécute plusieurs actions (pressions de touches) pendant une durée donnée."""
    # Appuyer sur toutes les touches spécifiées
    for action in actions:
        subprocess.Popen(['xdotool', 'keydown', action.value])

    time.sleep(duration)

    # Relâcher toutes les touches spécifiées
    for action in actions:
        subprocess.run(['xdotool', 'keyup', action.value])

def perform_actions():

    perform_action([Action.MOVE_RIGHT, Action.RUN], 1.8)
    perform_action([Action.MOVE_RIGHT, Action.JUMP, Action.RUN], 0.5)
    time.sleep(0.3)
    perform_action([Action.MOVE_RIGHT, Action.JUMP, Action.RUN], 0.05)
    perform_action([Action.MOVE_RIGHT, Action.RUN], 1)
    perform_action([Action.MOVE_RIGHT, Action.JUMP, Action.RUN], 1)


subprocess.Popen(['bash', '../SPMBros/buildproject_from_python.sh', 'build'])
time.sleep(2)
subprocess.run(['xdotool', 'keydown', 'Return'])
subprocess.run(['xdotool', 'keyup', 'Return'])

time.sleep(4)
perform_actions()