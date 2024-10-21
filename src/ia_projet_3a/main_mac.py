import subprocess
import time
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

def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    return client_socket

################### SCRIPT ###################

print("====================")
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
    print("##########################################")
   
    # Receive data
    data = client.recv(1024)

    if not data:
        break

    # read out data
    decoded_data = data.decode('utf-8').strip()
    print(decoded_data)

    # split data to get pos_x and pos_y
    data = decoded_data.split(',')

    if len(data) >= 14 and data[0] != '':

        x_pos, y_pos = map(int, data[:2])
        e1,e2,e3,e4,e5 = map(int, data [2:7])
        f1,f2,f3,f4,f5 = map(int, data [7:12])
        player_page = int(data[12])
        enemy_page = int(data[13])

        enemy_coord_X = 256 * enemy_page + e1
        player_coord_X = 256 * player_page + x_pos

        ecart = enemy_coord_X - player_coord_X
        if not f1:
            ecart = "null"

        print(f"========= PLAYER =========\n"
              f"Page : {player_page} | Xpos : {x_pos} | Xcoord : {player_coord_X}")
        print(f"========= ENEMY 1 =========\n"
              f"Existence : {f1} | Page : {enemy_page} | Xpos : {e1} | Xcoord : {enemy_coord_X}")
        print("========= MISC =========")
        print(f"Ecart entre player et enemy 1 : {ecart}")

        # print(x_pos,y_pos)
        # print(player_page)
        # print(enemy_page)
        # print(e1,e2,e3,e4,e5)
        # print(abs(e1 - x_pos))
        # print(f1 and abs(e1 - x_pos) < 50)
        # print(f1,f2,f3,f4,f5)

client.close()
print("Client died")
