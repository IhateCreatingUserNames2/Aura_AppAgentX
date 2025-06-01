# Backend Service Containerized Deployment

This project contains two containerized FastAPI services:

1. **Image Feature Extraction Service** (Port: 8001)
2. **Screen Parsing Service** (Port: 8000)

> Note: The Image Parsing Service is based on [OmniParser](https://github.com/microsoft/OmniParser), a comprehensive method for parsing user interface screenshots into structured elements developed by Microsoft.

## Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU support (NVIDIA Container Toolkit required)

## Directory Structure

```
.
├── docker-compose.yml
├── ImageEmbedding
│   ├── Dockerfile
│   ├── image_embedding.py
│   └── requirements.txt
└── OmniParser
    ├── Dockerfile
    ├── omni.py
    ├── requirements.txt
    ├── utils.py
    └── weights/
        ├── icon_detect_v1_5/
        └── icon_caption_florence/
```

## Model Weights Preparation

Before starting the services, you must download the required model weights from Hugging Face:

1. Visit [OmniParser on Hugging Face](https://huggingface.co/microsoft/OmniParser) to download all model files.
2. Place the model files in the following structure:
   - YOLO detection model weights at: `OmniParser/weights/icon_detect_v1_5/best.pt`
   - Caption model weights in: `OmniParser/weights/icon_caption_florence/` directory

**Note:** The service will not work properly without these model weights. Container building will proceed, but the API will fail to process images if weights are missing.

For more information about OmniParser, including usage examples, model details, and technical documentation, please refer to the [OmniParser official repository](https://github.com/microsoft/OmniParser).

## Building and Starting Services

```bash
# Build and start containers
docker-compose up --build

# Run in background
docker-compose up -d --build
```

## Service Access

- Image Feature Extraction Service: http://localhost:8001
- Image Parsing Service: http://localhost:8000

## API Documentation

### Image Feature Extraction Service

- `GET /available_models` - Get list of available models
- `POST /set_model` - Set the model to use
- `POST /extract_single/` - Extract features from a single image
- `POST /extract_batch/` - Batch extract features from multiple images
- `GET /model_info` - Get current model information
- `GET /benchmark/` - Run performance tests

### Image Parsing Service

- `POST /process_image/` - Process image and return parsing results

## Shutting Down Services

There are several ways to stop the Docker containers depending on your needs:

```bash
# Standard way to stop services and remove containers
docker-compose down

# Stop services but keep containers
docker-compose stop

# Stop services and remove containers, networks, and volumes
docker-compose down -v

# If you want to stop and remove everything including images
docker-compose down --rmi all -v
```

You can check the status of your containers using:

```bash
docker-compose ps
```

To view logs for troubleshooting:

```bash
# View logs from all services
docker-compose logs

# View logs from a specific service with follow option
docker-compose logs -f omniparser
``` 