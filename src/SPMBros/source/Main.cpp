#include <cstdio>
#include <iostream>

#include <SDL2/SDL.h>
#include "server.h"
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

uint8_t *romImage;
static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *texture;
static SDL_Texture *scanlineTexture;
static SMBEngine *smbEngine = nullptr;
static uint32_t renderBuffer[RENDER_WIDTH * RENDER_HEIGHT];

/**
 * Load the Super Mario Bros. ROM image.
 */
static bool loadRomImage()
{
    FILE *file = fopen(Configuration::getRomFileName().c_str(), "r");
    if (file == NULL)
    {
        std::cout << "Failed to open the file \"" << Configuration::getRomFileName() << "\". Exiting.\n";
        return false;
    }

    // Find the size of the file
    fseek(file, 0L, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0L, SEEK_SET);

    // Read the entire file into a buffer
    romImage = new uint8_t[fileSize];
    fread(romImage, sizeof(uint8_t), fileSize, file);
    fclose(file);

    return true;
}

/**
 * SDL Audio callback function.
 */
static void audioCallback(void *userdata, uint8_t *buffer, int len)
{
    if (smbEngine != nullptr)
    {
        smbEngine->audioCallback(buffer, len);
    }
}

/**
 * Initialize libraries for use.
 */
static bool initialize()
{
    // Load the configuration
    //
    Configuration::initialize(CONFIG_FILE_NAME);

    // Load the SMB ROM image
    if (!loadRomImage())
    {
        return false;
    }

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0)
    {
        std::cout << "SDL_Init() failed during initialize(): " << SDL_GetError() << std::endl;
        return false;
    }

    // Create the window
    window = SDL_CreateWindow(APP_TITLE,
                              SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED,
                              RENDER_WIDTH * Configuration::getRenderScale(),
                              RENDER_HEIGHT * Configuration::getRenderScale(),
                              0);
    if (window == nullptr)
    {
        std::cout << "SDL_CreateWindow() failed during initialize(): " << SDL_GetError() << std::endl;
        return false;
    }

    // Setup the renderer and texture buffer
    renderer = SDL_CreateRenderer(window, -1, (Configuration::getVsyncEnabled() ? SDL_RENDERER_PRESENTVSYNC : 0) | SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr)
    {
        std::cout << "SDL_CreateRenderer() failed during initialize(): " << SDL_GetError() << std::endl;
        return false;
    }

    if (SDL_RenderSetLogicalSize(renderer, RENDER_WIDTH, RENDER_HEIGHT) < 0)
    {
        std::cout << "SDL_RenderSetLogicalSize() failed during initialize(): " << SDL_GetError() << std::endl;
        return false;
    }

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, RENDER_WIDTH, RENDER_HEIGHT);
    if (texture == nullptr)
    {
        std::cout << "SDL_CreateTexture() failed during initialize(): " << SDL_GetError() << std::endl;
        return false;
    }

    if (Configuration::getScanlinesEnabled())
    {
        scanlineTexture = generateScanlineTexture(renderer);
    }

    // Set up custom palette, if configured
    //
    if (!Configuration::getPaletteFileName().empty())
    {
        const uint32_t *palette = loadPalette(Configuration::getPaletteFileName());
        if (palette)
        {
            paletteRGB = palette;
        }
    }

    if (Configuration::getAudioEnabled())
    {
        // Initialize audio
        SDL_AudioSpec desiredSpec;
        desiredSpec.freq = Configuration::getAudioFrequency();
        desiredSpec.format = AUDIO_S8;
        desiredSpec.channels = 1;
        desiredSpec.samples = 2048;
        desiredSpec.callback = audioCallback;
        desiredSpec.userdata = NULL;

        SDL_AudioSpec obtainedSpec;
        SDL_OpenAudio(&desiredSpec, &obtainedSpec);

        // Start playing audio
        SDL_PauseAudio(0);
    }

    if (!initializeServer())
    {
        return false;
    }

    return true;
}

static void shutdownSockets()
{
    if (clientSocket != -1)
    {
        std::cout << "Shutting down client socket" << std::endl;
        shutdown(clientSocket, SHUT_RDWR); // Interrompre envoi/réception
        close(clientSocket);               // Fermer le socket client
        clientSocket = -1;
    }

    if (serverSocket != -1)
    {
        std::cout << "Shutting down server socket" << std::endl;
        shutdown(serverSocket, SHUT_RDWR); // Interrompre envoi/réception
        close(serverSocket);               // Fermer le socket serveur
        serverSocket = -1;
    }
}

/**
 * Shutdown libraries for exit.
 */
static void shutdown()
{
    SDL_CloseAudio();

    SDL_DestroyTexture(scanlineTexture);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    shutdownSockets();

    SDL_Quit();
}

static void mainLoop()
{
    SMBEngine engine(romImage);
    smbEngine = &engine;
    engine.reset();

    bool running = true;
    int progStartTime = SDL_GetTicks();
    int frame = 0;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                switch (event.window.event)
                {
                case SDL_WINDOWEVENT_CLOSE:
                    running = false;
                    break;
                }
                break;

            default:
                break;
            }
        }

        const Uint8 *keys = SDL_GetKeyboardState(NULL);
        Controller &controller1 = engine.getController1();
        controller1.setButtonState(BUTTON_A, keys[SDL_SCANCODE_E]);
        controller1.setButtonState(BUTTON_B, keys[SDL_SCANCODE_R]);
        controller1.setButtonState(BUTTON_SELECT, keys[SDL_SCANCODE_T]);
        controller1.setButtonState(BUTTON_START, keys[SDL_SCANCODE_Y]);
        controller1.setButtonState(BUTTON_UP, keys[SDL_SCANCODE_UP]);
        controller1.setButtonState(BUTTON_DOWN, keys[SDL_SCANCODE_DOWN]);
        controller1.setButtonState(BUTTON_LEFT, keys[SDL_SCANCODE_LEFT]);
        controller1.setButtonState(BUTTON_RIGHT, keys[SDL_SCANCODE_RIGHT]);

        if (keys[SDL_SCANCODE_U])
        {
            // Reset
            engine.reset();
        }
        if (keys[SDL_SCANCODE_ESCAPE])
        {
            // Quit
            running = false;
            break;
        }
        if (keys[SDL_SCANCODE_F])
        {
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
        }

        // Récupérer la mémoire
        uint8_t xpos = engine.readFromMemory(0x86);
        uint8_t ypos = engine.readFromMemory(0xce);
        uint8_t player_page = engine.readFromMemory(0x6d);
        uint8_t player_X_speed = engine.readFromMemory(0x57);

        uint8_t enem_x_pos1 = engine.readFromMemory(0x87);
        uint8_t enem_x_pos2 = engine.readFromMemory(0x88);
        uint8_t enem_x_pos3 = engine.readFromMemory(0x89);
        uint8_t enem_x_pos4 = engine.readFromMemory(0x8a);
        uint8_t enem_x_pos5 = engine.readFromMemory(0x8b);

        uint8_t enem_flag1 = engine.readFromMemory(0x0f);
        uint8_t enem_flag2 = engine.readFromMemory(0x10);
        uint8_t enem_flag3 = engine.readFromMemory(0x11);
        uint8_t enem_flag4 = engine.readFromMemory(0x12);
        uint8_t enem_flag5 = engine.readFromMemory(0x13);

        uint8_t enemy_page1 = engine.readFromMemory(0x6e);
        uint8_t enemy_page2 = engine.readFromMemory(0x6f);
        uint8_t enemy_page3 = engine.readFromMemory(0x70);
        uint8_t enemy_page4 = engine.readFromMemory(0x71);
        uint8_t enemy_page5 = engine.readFromMemory(0x72);
        uint8_t enemy_page6 = engine.readFromMemory(0x73);

        //        uint8_t enem_ypos = engine.readFromMemory(0xcf);
        //        uint8_t enemy_type = engine.readFromMemory(0x16);
        //        uint8_t timer = engine.readFromMemory(0x0787);

        if (clientSocket != -1) // Si le client est connecté
        {
            // Formater les données en chaîne de caractères avec positions X et Y
            std::string data = std::to_string((int)xpos) + "," +
                               std::to_string((int)ypos) + "," +
                               std::to_string((int)player_page) + "," +
                               std::to_string((int)player_X_speed) + "," +
                               std::to_string((int)enem_x_pos1) + "," +
                               std::to_string((int)enem_x_pos2) + "," +
                               std::to_string((int)enem_x_pos3) + "," +
                               std::to_string((int)enem_x_pos4) + "," +
                               std::to_string((int)enem_x_pos5) + "," +
                               std::to_string((int)enem_flag1) + "," +
                               std::to_string((int)enem_flag2) + "," +
                               std::to_string((int)enem_flag3) + "," +
                               std::to_string((int)enem_flag4) + "," +
                               std::to_string((int)enem_flag5) + "," +
                               std::to_string((int)enemy_page1) + ",";

            // Envoyer les données
            send(clientSocket, data.c_str(), data.size(), 0);
        }

        engine.update();
        engine.render(renderBuffer);

        SDL_UpdateTexture(texture, NULL, renderBuffer, sizeof(uint32_t) * RENDER_WIDTH);

        SDL_RenderClear(renderer);

        // Render the screen
        SDL_RenderSetLogicalSize(renderer, RENDER_WIDTH, RENDER_HEIGHT);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Render scanlines
        if (Configuration::getScanlinesEnabled())
        {
            SDL_RenderSetLogicalSize(renderer, RENDER_WIDTH * 3, RENDER_HEIGHT * 3);
            SDL_RenderCopy(renderer, scanlineTexture, NULL, NULL);
        }

        SDL_RenderPresent(renderer);

        // Ensure that the framerate stays as close to the desired FPS as possible.
        int now = SDL_GetTicks();
        int delay = progStartTime + int(double(frame) * double(MS_PER_SEC) / double(Configuration::getFrameRate())) - now;
        if (delay > 0)
        {
            SDL_Delay(delay);
        }
        else
        {
            frame = 0;
            progStartTime = now;
        }
        frame++;
        SDL_Delay(0);
    }
}

int main(int argc, char **argv)
{
    if (!initialize())
    {
        std::cout << "Failed to initialize. Please check previous error messages for more information. The program will now exit.\n";
        return -1;
    }

    mainLoop();

    shutdown();

    return 0;
}
