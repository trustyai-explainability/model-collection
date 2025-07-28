# LMEval Assets Builder

This directory contains tools for building container images with pre-downloaded models and datasets for offline evaluation using LMEval.

## Overview

The LMEval Assets Builder automates the process of:
- Downloading models from Hugging Face Hub
- Downloading datasets for evaluation
- Building container images with the downloaded assets
- Pushing images to a container registry

## Prerequisites

- Python 3.8+
- Docker or Podman
- Make

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have Docker or Podman installed and configured.

## Configuration

The Makefile uses the following variables that can be customised:

- `MODEL`: The model to download (default: `google/flan-t5-base`)
- `IMAGE`: The base image name for the container registry (default: `quay.io/trustyai_testing/lmeval-assets`)
- `CONTAINER_ENGINE`: Container engine to use (default: `podman`, can be set to `docker`)

## Usage

### Available Targets

Run `make help` to see all available targets:

```bash
make help
```

### Building Assets

#### Build Flan Model with ARC-Easy Dataset
```bash
make assets-build-flan-arceasy
```

This will:
1. Delete existing data (with confirmation prompt)
2. Download the flan model
3. Download the ARC-Easy dataset
4. Build a container image with the assets

#### Build Flan Model with 20 Newsgroups Dataset
```bash
make assets-build-flan-20newsgroups
```

This will:
1. Delete existing data (with confirmation prompt)
2. Download the flan model
3. Download the 20 Newsgroups dataset
4. Build a container image with the assets

#### Push Images to Registry

After building, you can push the images to the registry:

```bash
# Push flan model with ARC-Easy dataset
make assets-push-flan-arceasy

# Push flan model with 20 Newsgroups dataset
make assets-push-flan-20newsgroups
```

### Using Different Container Engines

By default, the Makefile uses Podman. To use Docker instead:

```bash
make assets-build-flan-arceasy CONTAINER_ENGINE=docker
```

### Individual Tasks

You can also run individual tasks:

```bash
# Delete all data (with confirmation)
make delete-data

# Download the flan model
make download-flan

# Download ARC-Easy dataset
make download-dataset-arc-easy

# Download 20 Newsgroups dataset
make download-dataset-20newsgroups
```

## File Structure

- `Makefile`: Main build automation file
- `assets.Dockerfile`: Dockerfile for building container images
- `download_model.py`: Script for downloading models from Hugging Face Hub
- `download_dataset.py`: Script for downloading datasets
- `requirements.txt`: Python dependencies

## Container Images

The built images contain:
- Downloaded models in `/mnt/data/`
- Downloaded datasets in `/mnt/data/datasets/`
- Run as non-root user (UID 1000)

## Customisation

### Adding New Models

To add support for new models:

1. Add a new download target in the Makefile:
   ```makefile
   download-new-model:
       python download_model.py "model/name" downloads/new-model
       mkdir -p data/new-model
       cp -RL downloads/new-model/models--model--name/snapshots/*/* data/new-model/
   ```

2. Create a new build target:
   ```makefile
   assets-build-new-model: delete-data download-new-model download-dataset-arc-easy
       $(CONTAINER_ENGINE) build -t $(IMAGE)-new-model:latest -f assets.Dockerfile .
       $(CONTAINER_ENGINE) push $(IMAGE)-new-model:latest
   ```

### Adding New Datasets

To add support for new datasets:

1. Add a new download target in the Makefile:
   ```makefile
   download-dataset-new-dataset:
       python download_dataset.py "dataset/name" downloads/new-dataset
       mkdir -p data/datasets
       cp -R downloads/new-dataset/* data/datasets
   ```

2. Update existing build targets to include the new dataset dependency.

## Troubleshooting

### Permission Issues
If you encounter permission issues with Podman, ensure your user is configured for rootless containers or run with appropriate privileges.

### Network Issues
If downloads fail, check your internet connection and ensure you can access Hugging Face Hub.

### Storage Issues
Ensure you have sufficient disk space for downloading models and datasets, as they can be quite large.

## Contributing

When adding new features:
1. Update this README.md with new instructions
2. Add appropriate help text to the Makefile
3. Test with both Docker and Podman
4. Ensure all targets have proper dependencies 