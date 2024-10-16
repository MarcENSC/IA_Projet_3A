#ifndef SERVER_H
#define SERVER_H

#include <netinet/in.h>
#include <unistd.h>
#include <stdint.h>

// Declare variables and functions that need to be shared
extern  int serverSocket, clientSocket;

bool initializeServer();
void stopServer();
void sendGameData(int clientSocket, uint8_t xpos, uint8_t ypos);

#endif