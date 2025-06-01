# Aura 2: Advanced Conversational AI with Multi-Modal Capabilities

Aura 2 is a sophisticated conversational AI project integrating a core agent (Aura) with advanced memory systems (MemoryBlossom), Narrative Context Framing (NCF), and the ability to control Android applications via AppAgentX. This project showcases how these components can be orchestrated to create a rich, context-aware, and capable AI assistant.

Aura 2 is based on Aura and share it's core principles, read about Aura in :  https://github.com/IhateCreatingUserNames2/Aura 

**Core Features:**

*   **Aura Agent (Google ADK):** The primary conversational agent built using the Google Agent Development Kit (ADK).
*   **MemoryBlossom:** A custom long-term memory system designed for rich, interconnected memory storage and retrieval.
*   **Narrative Context Framing (NCF):** A prompting strategy that provides Aura with a deep contextual understanding of the conversation, including foundational narratives, RAG-enhanced information, and recent history.
*   **AppAgentX Integration:** Enables Aura to perceive and interact with Android GUIs, allowing it to perform tasks on a connected Android device or emulator (e.g., opening apps, learning new procedures). https://github.com/Westlake-AGI-Lab/AppAgentX 
*   **A2A Protocol Wrapper:** Exposes Aura via the Agent-to-Agent (A2A) communication protocol. 
*   ** AIRA HUB INTEGRATION READY **  https://github.com/IhateCreatingUserNames2/AiraHub2/ 
*   **Realtime Voice Interface (Prototype):** Includes a web-based client for real-time voice interaction, using OpenAI's Realtime API for STT/TTS and WebRTC.

## Hardware Usage 

- Aura uses 5 LLM calls for each AppAgentX Command 
- Aura was tested on a i7 7700, 16GB RAM, NVIDIA 1060 6GB, Windows 10 . It consumes at least 8gb of Ram and around 50gb of Space(In Windows due Docker/WSL...) 

## Project Structure


Aura2/



      ├── .env - Environment variables (API keys, configs)
      ├── a2a_wrapper/ - FastAPI server for A2A and Voice Client API
      │   ├── main.py - Main FastAPI application, A2A handler, Voice API endpoints
      │   ├── models.py - Pydantic models for A2A communication
      │   └── init.py
      ├── orchestrator_adk_agent.py - Core Aura ADK LlmAgent, tools, and session management
      ├── memory_system/ - Custom memory components
      │   ├── memory_blossom.py
      │   ├── memory_models.py
      │   ├── memory_connector.py
      │   ├── embedding_utils.py
      │   └── init.py
      ├── AppAgentX/ - AppAgentX submodule/package for Android control
      │   ├── backend/ - Dockerized backend services (OmniParser, etc.)
      │   │   ├── docker-compose.yml
      │   │   ├── OmniParser/
      │   │   └── ImageEmbedding/
      │   ├── config.py - AppAgentX specific configurations
      │   ├── deployment.py - Task execution logic
      │   ├── explor_auto.py - Automated exploration logic
      │   ├── ... (other AppAgentX files) ...
      │   └── init.py
      ├── aura_voice_client.html - HTML for realtime voice client
      ├── aura_voice_client.js - JavaScript for realtime voice client
      ├── index.html - HTML for A2A web chat UI
      ├── requirements.txt - Python dependencies
      └── README.md - Project documentation


## Architecture Overview

1.  **Client Interfaces:**
    *   **A2A Client (e.g., `index.html`):** Interacts with Aura via JSON-RPC calls to the `/` endpoint on the `a2a_wrapper` server.
    *   **Voice Client (`aura_voice_client.html`):** Uses WebRTC and OpenAI Realtime API for STT/TTS. Transcripts are sent to the `/aura/chat` endpoint, and text responses from Aura are sent back to OpenAI for TTS.

2.  **`a2a_wrapper/main.py` (FastAPI Server):**
    *   The main entry point and HTTP server.
    *   Hosts the A2A JSON-RPC endpoint (`/`).
    *   Hosts API endpoints for the voice client (`/openai/rt/session`, `/aura/chat`, `/health`).
    *   Implements Narrative Context Framing (NCF) pillar functions to build rich prompts.
    *   Uses `orchestrator_adk_agent.py` components (`adk_runner`, `orchestrator_adk_agent_aura`, `adk_session_service`) to process user requests.

3.  **`orchestrator_adk_agent.py` (Core Agent Logic):**
    *   Defines `orchestrator_adk_agent_aura` (an `LlmAgent`) with tools for memory, Android control (AppAgentX), etc.
    *   Manages ADK sessions using `InMemorySessionService`.
    *   Provides the `adk_runner` to execute agent turns.

4.  **`memory_system/`:**
    *   `MemoryBlossom`: Manages the storage, retrieval, and lifecycle of memories.
    *   `MemoryModels`: Defines the `Memory` Pydantic model.
    *   `MemoryConnector`: Builds relationships between memories.
    *   `EmbeddingUtils`: Handles text embeddings for memories.

5.  **`AppAgentX/` (Submodule/Package):**
    *   Contains the code for Android GUI interaction.
    *   Its `backend/` services (OmniParser, ImageEmbedding) run in Docker and provide screen parsing capabilities.
    *   Tools defined in `orchestrator_adk_agent.py` (`execute_android_task_tool_func`, `learn_android_procedure_tool_func`) call functions from `AppAgentX` (e.g., `appagentx_run_task_deployment`).

## Setup and Installation

### Prerequisites

1.  **Python:** Version 3.10+ recommended.
2.  **Pip:** Python package installer.
3.  **Virtual Environment:** Recommended (e.g., `venv`).
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate    # Windows
    ```
4.  **Docker & Docker Compose:** Required for running AppAgentX backend services. Ensure Docker Desktop is installed, running, and configured correctly (e.g., WSL2 backend on Windows, sufficient resources allocated).
5.  **Android Device or Emulator:** With USB Debugging enabled and authorized.
6.  **ADB (Android Debug Bridge):** Must be installed and in your system's `PATH`.
7.  **Git LFS:** (If AppAgentX or its dependencies use it for large files)
    ```bash
    git lfs install
    ```

### Installation Steps

1.  **Clone the Repository (if applicable):**
    ```bash
     Clone download this repository 
    ```
    (If AppAgentX is a submodule, initialize and update it: `git submodule update --init --recursive`)

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt ( DEPRECATED, nEED TO BE ADJUSTED, YOU MAY HAVE TO MANUALLY INSTALL SOME STUFF FOR NOW, CHECK FILES) 
    ```
    (There might be a separate `requirements.txt` inside `AppAgentX/` that its components need. Check AppAgentX setup instructions.)

3.  **Set up Environment Variables:**
    *   Create a `.env` file in the project root (`Aura2/.env`).
    *   Populate it with necessary API keys and configurations. See `.env.example` (if you create one) or refer to the required variables below:
        ```env
        # Aura2 Main .env
        OPENROUTER_API_KEY="sk-or-v1-..." # For Aura's main LLM via OpenRouter
        OPENAI_API_KEY="sk-..."          # For OpenAI services (Realtime STT/TTS, potentially AppAgentX LLM)
        AURA_MODEL_IDENTIFIER="openrouter/openai/gpt-4o-mini" # Or your preferred OpenRouter model for Aura

        # A2A Wrapper Config (defaults are usually fine for local)
        # A2A_WRAPPER_HOST="localhost"
        # A2A_WRAPPER_PORT="8094"
        # A2A_WRAPPER_BASE_URL="http://localhost:8094" # Change if using ngrok for public access

        # MemoryBlossom
        MEMORY_BLOSSOM_PERSISTENCE_PATH="a2a_wrapper/memory_blossom_data.json" # Or your preferred path

        # AppAgentX Backend Service URIs (if not hardcoded in AppAgentX/config.py)
        APPAGENTX_OMNI_URI="http://127.0.0.1:8000"    # Default if OmniParser Docker maps to host port 8000
        APPAGENTX_FEATURE_URI="http://127.0.0.1:8001" # Default for ImageEmbedding Docker

        # AppAgentX Internal LLM (if it uses its own separate from Aura's main LLM)
        APPAGENTX_LLM_API_KEY="sk-..." # e.g., another OpenAI key or DeepSeek key
        APPAGENTX_LLM_MODEL="gpt-4o"   # Or model used by AppAgentX

        # AppAgentX Database Config (if not hardcoded in AppAgentX/config.py)
        # APPAGENTX_NEO4J_URI="neo4j+s://<your_neo4j_instance>.databases.neo4j.io"
        # APPAGENTX_NEO4J_AUTH_USER="neo4j"
        # APPAGENTX_NEO4J_AUTH_PASS="<your_neo4j_password>"
        # APPAGENTX_PINECONE_API_KEY="..."
        # APPAGENTX_PINECONE_ENVIRONMENT="..."
        ```

4.  **Set up AppAgentX Backend Services (Docker):**
    *   **Download Model Weights:** Follow the instructions in `AppAgentX/backend/README.md` to download the necessary model weights (e.g., for OmniParser's icon detection and captioning) and place them in the correct subdirectories within `AppAgentX/backend/OmniParser/weights/`.
    *   **Navigate to the backend directory:**
        ```bash
        cd AppAgentX/backend
        ```
    *   **Build and start the Docker containers:**
        ```bash
        docker-compose up --build -d
        ```
    *   Verify containers are running: `docker ps` and check logs:
        ```bash
        docker-compose logs -f omni-parser 
        docker-compose logs -f image-embedding 
        ```
        (Use the correct service names from your `docker-compose.yml`). Ensure they start without errors and are listening on their respective ports (e.g., 8000 and 8001).

5.  **Prepare Android Device/Emulator:**   ( DEFAULT DEVICE_ID = DYONHQPZ9PF6V4TO <- FIND THIS IN THE CODE FILES (orchestrator...) and REPLACE WITH YOUR OWN DEVICE ID ) 
    *   Connect your Android device via USB or start an emulator.
    *   Enable Developer Options and **USB Debugging**.
    *   **Crucially, enable any "Security Settings" related to USB Debugging or "Disable permission monitoring"** in Developer Options to allow ADB to inject input events (taps, swipes). This is device-manufacturer specific.
    *   Authorize your computer for USB debugging when prompted on the device.
    *   Verify connection: `adb devices` (should show your device as `device`).

## Running Aura2

1.  **Ensure AppAgentX backend Docker containers are running.**
2.  **Ensure your Android device/emulator is connected and authorized for ADB.**
3.  **Start the Aura2 main server (FastAPI):**
    From the project root (`Aura2/`):
    ```bash
    python a2a_wrapper/main.py
    ```
    The server should start, typically on `http://localhost:8094`.

## Using the Interfaces

*   **A2A Web Chat UI:**
    *   Open `index.html` (located in the `Aura2/` root) in your web browser.
    *   Interact with Aura via text.

*   **Realtime Voice Client:** ( NOT IMPLEMENTED IN THE CURRENT LOGIC, RealTime Works but wont run thru AURA FLOW) 
    *   Open `aura_voice_client.html` (located in the `Aura2/` root) in your web browser.
    *   Click "Connect to Realtime Services."
    *   Allow microphone access when prompted.
    *   Use "Start Speaking" and "Stop Speaking" to talk to Aura.

## Development & Debugging

*   **Server Logs:** Monitor the terminal where `a2a_wrapper/main.py` is running for detailed logs from Aura, ADK, NCF pillars, and AppAgentX tool calls.
*   **AppAgentX Backend Logs:** Use `docker-compose logs -f <service_name>` in the `AppAgentX/backend/` directory.
*   **Browser Developer Console:** Check for JavaScript errors or network request issues for both web UIs.
*   **ADB:** Use `adb logcat` or specific `adb shell` commands to debug device interactions.

## Key Files for Modification/Understanding

*   **`a2a_wrapper/main.py`:** Main server logic, NCF pillars, API endpoint definitions.
*   **`orchestrator_adk_agent.py`:** Core ADK `LlmAgent` definition, tool function implementations (which call AppAgentX).
*   **`AppAgentX/config.py`:** Configuration for AppAgentX components (LLM, DBs, service URIs).
*   **`AppAgentX/deployment.py`, `AppAgentX/explor_auto.py`:** Core logic for AppAgentX operations.
*   **`AppAgentX/backend/docker-compose.yml`:** Defines the backend microservices.
*   **`.env`:** For all your secrets and environment-specific configurations.

## Troubleshooting Common Issues

*   **`ModuleNotFoundError`:**
    *   Ensure your virtual environment is activated.
    *   Check that `Aura2/` project root is in `sys.path` (handled by scripts).
    *   Ensure all internal imports within the `AppAgentX` package use correct explicit relative imports (`from .module` or `from ..module`).
*   **Docker Container Failures (OmniParser, ImageEmbedding):**
    *   Check `docker-compose logs -f <service_name>`.
    *   Ensure model weights are downloaded and placed correctly as per `AppAgentX/backend/README.md`.
    *   Verify `requirements.txt` within the Docker context for version compatibility (e.g., `transformers` library version for Florence-2).
    *   Check for port conflicts on your host machine.
    *   Ensure Docker Desktop has sufficient resources and NVIDIA GPU support is working if required by containers.
*   **ADB Errors (`'adb' not recognized`, `INJECT_EVENTS permission`):**
    *   Ensure ADB platform-tools directory is in your system `PATH` and your IDE/terminal has picked up the change.
    *   On the Android device, enable "USB Debugging (Security Settings)" or "Disable permission monitoring" in Developer Options. Re-authorize USB debugging.
*   **`ERR_EMPTY_RESPONSE` or `RemoteDisconnected` when AppAgentX calls its backend:**
    *   Verify the relevant Docker container (e.g., `omni-parser`) is running and its internal web server has started successfully (check container logs).
    *   Ensure `APPAGENTX_OMNI_URI` / `APPAGENTX_FEATURE_URI` in your `.env` file (and used by AppAgentX) point to the correct host ports mapped in `docker-compose.yml`.

## Future Enhancements
*   [TODO: List potential future work or improvements]
*   More robust error handling in AppAgentX tool calls.
*   Persistent caching for downloaded AppAgentX models in Docker volumes.
*   UI improvements for voice and A2A clients.
*   STT and TTS integration with RealTime Models and Aura ADK Orchestrator

---

Good luck!




![image](https://github.com/user-attachments/assets/6476c98f-e418-4205-8843-667d302ce36a)

