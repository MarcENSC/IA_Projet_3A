import time
import subprocess
import torch
from game_manager import controls, environment, map_manager
from game_manager.actions import Action
from ai import neural_network
from network import server, data_parser

def main():
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
    time.sleep(2)

    # Connexion au serveur
    client = server.connect_to_server()

    while True:
        # Recevoir les données du serveur
        data = client.recv(4096)
        if not data:
            break

    
    client.close()

if __name__ == "__main__":
    main()
