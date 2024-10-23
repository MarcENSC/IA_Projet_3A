import subprocess
import time
from enum import Enum
from methods import *

class Action(Enum):
    MOVE_RIGHT = 'Right'
    MOVE_LEFT = 'Left'
    CROUCH = 'Down'
    RUN = 'R'
    JUMP = 'E'
    ENTER = 'Y'
    ESCAPE = 'T'

################### SCRIPT ###################

# Choose os : "linux", "mac"
# used_os = "mac"

print("====================\nWaiting for game/server")
subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
time.sleep(2)
print("Game/Server launched")

print("====================\nWaiting for client")
client = connect_to_server()
print("Client launched")

floors, stocks = extract_floors_and_stocks('../maps/World11.json')

print("====================\nClient listening...")

while True:
    print(2*"##########################################\n")
   
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
        print(f"Au dessus du sol : {is_above_floor(player_coord_X/2,floors)}")


    print("\n")

client.close()
print("Client died")
