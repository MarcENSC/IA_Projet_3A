import subprocess
import time
import socket
from enum import Enum

# Choose os : "linux", "mac"
os = "mac"

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
    Args:
    actions: List of actions from Action enum (e.g., [Action.RUN, Action.MOVE_RIGHT])
    duration: duration of the action (e.g., 3 for 3 seconds)
    """
    for action in actions:
        subprocess.Popen(['xdotool', 'keydown', action.value])

    time.sleep(duration)

    # Key up for each action
    for action in actions:
        subprocess.run(['xdotool', 'keyup', action.value])

def press(action):
    subprocess.Popen(['xdotool', 'keydown', action.value])

def unpress(action):
    subprocess.Popen(['xdotool', 'keyup', action.value])


def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    return client_socket


################### SCRIPT ###################

print("====================")
print("Waiting for game/server")

subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
time.sleep(2)
print("Game/Server launched")

print("====================")
print("Waiting for client")
client = connect_to_server()
print("Client launched")

print("====================")
print("Client listening...")

while True:
    print("##########################################")
   
    # Receive data
    data = client.recv(1024)

    if not data:
        break

    # read out data
    decoded_data = data.decode('utf-8').strip()
    print(f"========= RECEIVED DATA =========")
    print(decoded_data)

    # split data to get pos_x and pos_y
    data = decoded_data.split(',')

    if len(data) >= 14 and data[0] != '':

        # Name data
        x_pos, y_pos = map(int, data[:2])
        e1,e2,e3,e4,e5 = map(int, data [2:7])
        f1,f2,f3,f4,f5 = map(int, data [7:12])
        player_page = int(data[12])
        enemy_page = int(data[13])

        # Determine real coords
        enemy_coord_X = 256 * enemy_page + e1
        player_coord_X = 256 * player_page + x_pos

        # Get gap
        ecart = enemy_coord_X - player_coord_X
        if not f1:
            ecart = "null"

        print(f"========= PLAYER =========\n"
              f"Page : {player_page} | Xpos : {x_pos} | Xcoord : {player_coord_X}")
        print(f"========= ENEMY 1 =========\n"
              f"Existence : {f1} | Page : {enemy_page} | Xpos : {e1} | Xcoord : {enemy_coord_X}")
        print("========= MISC =========")
        print(f"Ecart entre player et enemy 1 : {ecart}")

client.close()
print("Client died")
