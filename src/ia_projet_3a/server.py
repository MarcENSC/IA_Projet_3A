import socket
import struct

def start_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the server address and port
    server_address = ('localhost', 8080)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print("Server is listening for connections...")

    while True:
        try:
            # Wait for a connection
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address} established.")

            while True:
                # Receive data from the client
                data = client_socket.recv(2)  # Receive 2 bytes for x and y positions

                if not data:
                    print("Connection closed by the client.")
                    break

                if len(data) == 2:
                    # Unpack the received bytes into integers
                    x_position, y_position = struct.unpack('BB', data)
                    print(f"Received Position - X: {x_position}, Y: {y_position}")
                else:
                    print(f"Received unexpected number of bytes: {len(data)}")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the client socket
            client_socket.close()
            print("Client connection closed.")


