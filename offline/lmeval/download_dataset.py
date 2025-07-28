import sys
import os
from datasets import load_dataset


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: download.py <dataset_name> <destination_path> [config]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    destination_path = sys.argv[2]
    config = sys.argv[3] if len(sys.argv) == 4 else "default"

    os.makedirs(destination_path, exist_ok=True)

    print(
        f"Downloading dataset '{dataset_name}' with config '{config}' to '{destination_path}'..."
    )

    dataset = load_dataset(
        dataset_name, config, cache_dir=destination_path, trust_remote_code=True
    )

    print(
        f"Dataset '{dataset_name}' with config '{config}' downloaded successfully to '{destination_path}'."
    )
    print("Dataset download complete.")


if __name__ == "__main__":
    main()
