import socket

def main():
    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect(('127.0.0.1', 8080))

    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(data.decode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    main()
