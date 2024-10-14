import subprocess
import time
from enum import Enum
import socket

class Action(Enum):
    MOVE_RIGHT = 'Right'
    MOVE_LEFT = 'Left'
    CROUCH = 'Down'
    RUN = 'R'
    JUMP = 'E'
    ENTER = 'Y'
    ESCAPE = 'T'

def perform_action(actions, duration):
    """
    Args :
    actions : List of actions from Action enum (ex : [Action.RUN, Action.MOVE_RIGHT])
    duration : duration of the action (ex: 3 for 3sec)
    """

    for action in actions:
        subprocess.Popen(['xdotool', 'keydown', action.value])

    time.sleep(duration)

    for action in actions:
        subprocess.run(['xdotool', 'keyup', action.value])

def connect_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    return client_socket

################### SCRIPT ###################
print("Waiting for game/server")
subprocess.Popen(['bash', '../SPMBros/buildproject_from_python.sh', 'build'])
print("Game/Server launched")

time.sleep(2)

print("Waiting for client")
client = connect_server()
print("Client launched")

print("Client active")
while True:
    data = client.recv(1024)
    if not data:
        break
    print(data.decode('utf-8'))
client.close()
print("Client died")