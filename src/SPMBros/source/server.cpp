#include <cstdio>
#include <iostream>

#include <SDL2/SDL.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

#include "Emulation/Controller.hpp"
#include "SMB/SMBEngine.hpp"
#include "Util/Video.hpp"

#include "Configuration.hpp"
#include "Constants.hpp"

#include "Emulation/MemoryAccess.hpp"
int serverSocket;
int clientSocket;
bool initializeServer()
{
    
    // Créer le socket
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
    {
        perror("Erreur de création du socket");
        return false;
    }

    // Configurer l'adresse du serveur
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY; // Accepter toutes les connexions
    serverAddr.sin_port = htons(8080); // Port 8080

    // Lier le socket
    if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
    {
        perror("Erreur de liaison du socket");
        close(serverSocket); // Fermer le socket en cas d'erreur
        return false;
    }

    // Écouter les connexions
    if (listen(serverSocket, 1) < 0)
    {
        perror("Erreur d'écoute sur le socket");
        close(serverSocket); // Fermer le socket en cas d'erreur
        return false;
    }

    // Accepter une connexion
    clientSocket = accept(serverSocket, nullptr, nullptr);
    if (clientSocket < 0)
    {
        perror("Erreur d'acceptation de la connexion");
        close(serverSocket); // Fermer le socket en cas d'erreur
        return false;
    }

    return true;
}