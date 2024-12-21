import time
import subprocess
from network import server, data_parser
from game_manager import map_manager

def main():
    subprocess.Popen(['bash', "../SPMBros/buildproject_from_python.sh", 'build'])
    time.sleep(2)

    filename = 'maps/World11.json'
    map_matrix = map_manager.parse_json_to_matrix(filename)

    # Connexion au serveur
    client = server.connect_to_server()

    while True:
        # Recevoir les données du serveur
        data = client.recv(4096)
        if not data:
            break
        
        print("\n\n\n\n\n")
        data_parser.parse_game_data(data,map_matrix)

    
    client.close()

if __name__ == "__main__":
    main()
