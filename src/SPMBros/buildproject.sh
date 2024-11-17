#!/bin/bash

# Fonction pour construire le projet
build_project() {
    echo "Creating build directory and building the project..."
    mkdir -p build
    cd build
    
    # Création et configuration du fichier smbc.conf
    echo "[audio]" > smbc.conf
    echo "frequency = 22050" >> smbc.conf
    echo "[game]" >> smbc.conf
    echo "rom_file = ../Super Mario Bros. (JU) (PRG0) [!].nes" >> smbc.conf
    
    # Exécution de CMake et compilation du projet
    cmake ..
    make
    cd ..
}

# Vérification de l'argument
if [ "$1" == "build" ]; then
    if [ -d "build" ]; then
        echo "Build directory exists. Running smbc..."
        cd build
        ./smbc
        cd ..
    else
        build_project
    fi
elif [ "$1" == "rebuild" ]; then
    if [ -d "build" ]; then
        echo "Rebuilding the project..."
        rm -rf build
    fi
    build_project
else
    echo "Usage: $0 {build|rebuild}"
    exit 1
fi
