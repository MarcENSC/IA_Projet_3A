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

def press(action):
    subprocess.Popen(['xdotool', 'keydown', action.value])

def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    return client_socket


################### SCRIPT ###################

print("Waiting for game/server")
subprocess.Popen(['bash', '../SPMBros/buildproject_from_python.sh', 'build'])
time.sleep(2)
print("Game/Server launched")

print("====================")

print("Waiting for client")
client = connect_to_server()
print("Client launched")

print("====================")

time.sleep(1)

print("Starting game...")
perform_action([Action.ENTER],1)
time.sleep(3)

print("====================")

print("Client listening...")

# press(Action.MOVE_RIGHT)

while True:
    # Receive data
    data = client.recv(1024)

    if not data:
        break

    # read out data
    decoded_data = data.decode('utf-8').strip()

    # split data to get pos_x and pos_y
    positions = decoded_data.split(',')

    if len(positions) >= 2:
        x_pos, y_pos = map(int, positions[:2])
        print(x_pos,y_pos)
        # if abs(enemy_x_pos - x_pos) < 10:
        #     press(Action.JUMP)

client.close()
print("Client died")
