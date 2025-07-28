# Variables
MODEL := google/flan-t5-base
IMAGE := quay.io/trustyai_testing/lmeval-assets
CONTAINER_ENGINE ?= podman

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  assets-build-flan-arceasy     - Build flan model with ARC-Easy dataset"
	@echo "  assets-build-flan-20newsgroups - Build flan model with 20 Newsgroups dataset"
	@echo "  delete-data                   - Remove all data"
	@echo "  download-flan                 - Download flan model"
	@echo "  download-dataset-arc-easy     - Download ARC-Easy dataset"
	@echo "  download-dataset-20newsgroups - Download 20 Newsgroups dataset"

# Delete all data
.PHONY: delete-data
delete-data:
	@echo "This will delete all data in the data/ directory."
	@read -p "Are you sure you want to continue? (y/N): " confirm && [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ] || (echo "Operation cancelled." && exit 1)
	rm -Rf data/*

# Download flan model
.PHONY: download-flan
download-flan:
	python download_model.py "google/flan-t5-base" downloads/flan
	mkdir -p data/flan
	cp -RL downloads/flan/models--google--flan-t5-base/snapshots/*/* data/flan/

# Download ARC-Easy dataset
.PHONY: download-dataset-arc-easy
download-dataset-arc-easy:
	python download_dataset.py "allenai/ai2_arc" downloads/arc_easy "ARC-Easy"
	mkdir -p data/datasets
	cp -R downloads/arc_easy/* data/datasets

# Download 20 Newsgroups dataset
.PHONY: download-dataset-20newsgroups
download-dataset-20newsgroups:
	python download_dataset.py "xwjzds/20_newsgroups" downloads/20newsgroups
	mkdir -p data/datasets
	cp -R downloads/20newsgroups/* data/datasets

# Build flan model with ARC-Easy dataset
.PHONY: assets-build-flan-arceasy
assets-build-flan-arceasy: delete-data download-flan download-dataset-arc-easy
	$(CONTAINER_ENGINE) build -t $(IMAGE)-flan-arceasy:latest -f assets.Dockerfile .
	$(CONTAINER_ENGINE) push $(IMAGE)-flan-arceasy:latest

# Build flan model with 20 Newsgroups dataset
.PHONY: assets-build-flan-20newsgroups
assets-build-flan-20newsgroups: delete-data download-flan download-dataset-20newsgroups
	$(CONTAINER_ENGINE) build -t $(IMAGE)-flan-20newsgroups:latest -f assets.Dockerfile .
	$(CONTAINER_ENGINE) push $(IMAGE)-flan-20newsgroups:latest 