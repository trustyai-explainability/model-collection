import sys
import os
from huggingface_hub import snapshot_download


def main():
    if len(sys.argv) != 3:
        print("Usage: download.py <model_name> <destination_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    destination_path = sys.argv[2]


    os.makedirs(destination_path, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{destination_path}'...")


    snapshot_download(
        repo_id=model_name,
        repo_type="model",
        cache_dir=destination_path,
    )

    print(f"Model '{model_name}' downloaded successfully to '{destination_path}'.")


if __name__ == "__main__":
    main()
