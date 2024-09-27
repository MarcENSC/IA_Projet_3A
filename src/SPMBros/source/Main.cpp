#include <cstdio>
#include <iostream>

#include <SDL2/SDL.h>

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

    return true;
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
        controller1.setButtonState(BUTTON_A, keys[SDL_SCANCODE_X]);
        controller1.setButtonState(BUTTON_B, keys[SDL_SCANCODE_Z]);
        controller1.setButtonState(BUTTON_SELECT, keys[SDL_SCANCODE_BACKSPACE]);
        controller1.setButtonState(BUTTON_START, keys[SDL_SCANCODE_RETURN]);
        controller1.setButtonState(BUTTON_UP, keys[SDL_SCANCODE_UP]);
        controller1.setButtonState(BUTTON_DOWN, keys[SDL_SCANCODE_DOWN]);
        controller1.setButtonState(BUTTON_LEFT, keys[SDL_SCANCODE_LEFT]);
        controller1.setButtonState(BUTTON_RIGHT, keys[SDL_SCANCODE_RIGHT]);

        if (keys[SDL_SCANCODE_R])
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

        // Récupérer la mémoire à l'adresse 0x86
        uint8_t xpos = engine.readFromMemory(0x86);
        uint8_t ypos = engine.readFromMemory(0xce);
        uint8_t enem_xpos = engine.readFromMemory(0x87);
        uint8_t enem_ypos = engine.readFromMemory(0xcf);
        uint8_t enemy_type = engine.readFromMemory(0x16);
        uint8_t timer = engine.readFromMemory(0x0787);
        uint8_t test = engine.readFromMemory(0xa5);

        std::cout << "===========================" << std::endl;
        std::cout << "Player X position: " << (int)xpos << std::endl;
        std::cout << "Player Y position: " << (int)ypos << std::endl;
        std::cout << "Enemy X position: " << (int)enem_xpos << std::endl;
        std::cout << "Enemy Y position: " << (int)enem_ypos << std::endl;
        std::cout << "enemy type: " << (int)enemy_type << std::endl;
        std::cout << "Timer frame ?: " << (int)timer << std::endl;
        std::cout << "Test: " << (int)test << std::endl;

//        uint8_t data_1 = engine.readFromMemory(0x76);
//        std::cout << "Data 1: " << (int)data_1 << std::endl;
//
//        uint8_t data_2 = engine.readFromMemory(0xdd);
//        std::cout << "Data 2: " << (int)data_2 << std::endl;
//
//        uint8_t data_3 = engine.readFromMemory(0xbb);
//        std::cout << "Data 3: " << (int)data_3 << std::endl;
//
//        uint8_t data_4 = engine.readFromMemory(0x4c);
//        std::cout << "Data 4: " << (int)data_4 << std::endl;
//
//        uint8_t data_5 = engine.readFromMemory(0xea);
//        std::cout << "Data 5: " << (int)data_5 << std::endl;
//
//        uint8_t data_6 = engine.readFromMemory(0x1d);
//        std::cout << "Data 6: " << (int)data_6 << std::endl;
//
//        uint8_t data_7 = engine.readFromMemory(0x1b);
//        std::cout << "Data 7: " << (int)data_7 << std::endl;
//
//        uint8_t data_8 = engine.readFromMemory(0xcc);
//        std::cout << "Data 8: " << (int)data_8 << std::endl;
//
//        uint8_t data_9 = engine.readFromMemory(0x56);
//        std::cout << "Data 9: " << (int)data_9 << std::endl;
//
//        uint8_t data_10 = engine.readFromMemory(0x5d);
//        std::cout << "Data 10: " << (int)data_10 << std::endl;
//
//        uint8_t data_11 = engine.readFromMemory(0x16);
//        std::cout << "Data 11: " << (int)data_11 << std::endl;
//
//        uint8_t data_12 = engine.readFromMemory(0x9d);
//        std::cout << "Data 12: " << (int)data_12 << std::endl;
//
//        uint8_t data_13 = engine.readFromMemory(0xc6);
//        std::cout << "Data 13: " << (int)data_13 << std::endl;
//
//        uint8_t data_14 = engine.readFromMemory(0x1d);
//        std::cout << "Data 14: " << (int)data_14 << std::endl;
//
//        uint8_t data_15 = engine.readFromMemory(0x36);
//        std::cout << "Data 15: " << (int)data_15 << std::endl;
//
//        uint8_t data_16 = engine.readFromMemory(0x9d);
//        std::cout << "Data 16: " << (int)data_16 << std::endl;
//
//        uint8_t data_17 = engine.readFromMemory(0xc9);
//        std::cout << "Data 17: " << (int)data_17 << std::endl;
//
//        uint8_t data_18 = engine.readFromMemory(0x1d);
//        std::cout << "Data 18: " << (int)data_18 << std::endl;
//
//        uint8_t data_19 = engine.readFromMemory(0x04);
//        std::cout << "Data 19: " << (int)data_19 << std::endl;
//
//        uint8_t data_20 = engine.readFromMemory(0xdb);
//        std::cout << "Data 20: " << (int)data_20 << std::endl;
//
//        uint8_t data_21 = engine.readFromMemory(0x49);
//        std::cout << "Data 21: " << (int)data_21 << std::endl;
//
//        uint8_t data_22 = engine.readFromMemory(0x1d);
//        std::cout << "Data 22: " << (int)data_22 << std::endl;
//
//        uint8_t data_23 = engine.readFromMemory(0x84);
//        std::cout << "Data 23: " << (int)data_23 << std::endl;
//
//        uint8_t data_24 = engine.readFromMemory(0x1b);
//        std::cout << "Data 24: " << (int)data_24 << std::endl;
//
//        uint8_t data_25 = engine.readFromMemory(0xc9);
//        std::cout << "Data 25: " << (int)data_25 << std::endl;
//
//        uint8_t data_26 = engine.readFromMemory(0x5d);
//        std::cout << "Data 26: " << (int)data_26 << std::endl;
//
//        uint8_t data_27 = engine.readFromMemory(0x88);
//        std::cout << "Data 27: " << (int)data_27 << std::endl;
//
//        uint8_t data_28 = engine.readFromMemory(0x95);
//        std::cout << "Data 28: " << (int)data_28 << std::endl;
//
//        uint8_t data_29 = engine.readFromMemory(0x0f);
//        std::cout << "Data 29: " << (int)data_29 << std::endl;
//
//        uint8_t data_30 = engine.readFromMemory(0x08);
//        std::cout << "Data 30: " << (int)data_30 << std::endl;
//
//        uint8_t data_31 = engine.readFromMemory(0x30);
//        std::cout << "Data 31: " << (int)data_31 << std::endl;
//
//        uint8_t data_32 = engine.readFromMemory(0x4c);
//        std::cout << "Data 32: " << (int)data_32 << std::endl;
//
//        uint8_t data_33 = engine.readFromMemory(0x78);
//        std::cout << "Data 33: " << (int)data_33 << std::endl;
//
//        uint8_t data_34 = engine.readFromMemory(0x2d);
//        std::cout << "Data 34: " << (int)data_34 << std::endl;
//
//        uint8_t data_35 = engine.readFromMemory(0xa6);
//        std::cout << "Data 35: " << (int)data_35 << std::endl;
//
//        uint8_t data_36 = engine.readFromMemory(0x28);
//        std::cout << "Data 36: " << (int)data_36 << std::endl;
//
//        uint8_t data_37 = engine.readFromMemory(0x90);
//        std::cout << "Data 37: " << (int)data_37 << std::endl;
//
//        uint8_t data_38 = engine.readFromMemory(0xb5);
//        std::cout << "Data 38: " << (int)data_38 << std::endl;
//
//        uint8_t data_39 = engine.readFromMemory(0xff);
//        std::cout << "Data 39: " << (int)data_39 << std::endl;

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
