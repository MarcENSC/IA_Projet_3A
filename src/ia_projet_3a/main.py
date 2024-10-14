import subprocess
import time
import struct
import socket
from enum import Enum

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
    # Key down for each action
    for action in actions:
        subprocess.Popen(['xdotool', 'keydown', action.value])

    time.sleep(duration)

    # Key up for each action
    for action in actions:
        subprocess.run(['xdotool', 'keyup', action.value])

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

print("Client listening...")

while True:
    data = bytearray()  # Initialize a bytearray to accumulate data
    while len(data) < 2:  # Ensure we receive exactly 2 bytes
        packet = client.recv(2 - len(data))  # Receive the remaining bytes needed
        if not packet:
            break
        data.extend(packet)

    if len(data) != 2:
        print(f"Unexpected data length: {len(data)}")
        break

    x_position, y_position = struct.unpack('BB', data)
    print(f"Received Position - X: {x_position}, Y: {y_position}")

    # Check the X position to perform an action
    if x_position == 90:  # Adjust this condition as needed
        perform_action([Action.JUMP], 1)
client.close()
print("Client died")
