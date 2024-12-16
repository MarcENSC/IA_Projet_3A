import socket
import time

def connect_to_server(host='127.0.0.1', port=8080):
    """ Se connecter au serveur du jeu """
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))  # Connexion au serveur local
        return client_socket
    except:
        return False
