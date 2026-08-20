from huggingface_hub import snapshot_download, HfApi
import json
import os
from datetime import datetime


BASE_DIR = "models"
INVENTORY_FILE = "inventory.json"


def get_model_location(repo_id):
    """
    Hugging Face format:
    
    organization/model_name

    Example:
    deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

    Result:
    models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    """

    parts = repo_id.split("/", 1)

    if len(parts) != 2:
        raise ValueError(
            "Invalid Hugging Face repo format. "
            "Expected: organization/model_name"
        )

    organization = parts[0]
    model_name = parts[1]

    path = os.path.join(
        BASE_DIR,
        organization,
        model_name
    )

    return organization, model_name, path



def get_model_metadata(repo_id):

    api = HfApi()

    try:
        info = api.model_info(repo_id)

        license_name = "Unknown"

        if info.cardData:
            license_name = info.cardData.get(
                "license",
                "Unknown"
            )

        return {
            "author": info.author,
            "license": license_name,
            "downloads": info.downloads
        }

    except Exception:

        return {
            "author": "Unknown",
            "license": "Unknown",
            "downloads": 0
        }



def update_inventory(record):

    if os.path.exists(INVENTORY_FILE):

        with open(
            INVENTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            inventory = json.load(f)

    else:

        inventory = []


    inventory.append(record)


    with open(
        INVENTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            inventory,
            f,
            indent=4,
            ensure_ascii=False
        )



def download_model(repo_id):

    organization, model_name, path = get_model_location(
        repo_id
    )


    print("\n================================")
    print("MODEL DOWNLOAD")
    print("================================")

    print(f"\nRepository:")
    print(repo_id)

    print(f"\nOrganization:")
    print(organization)

    print(f"\nModel:")
    print(model_name)

    print(f"\nSaving location:")
    print(path)


    confirm = input(
        "\nContinue download? (yes/no): "
    )


    if confirm.lower() != "yes":

        print("\nCancelled.")
        return



    os.makedirs(
        path,
        exist_ok=True
    )


    print("\nDownloading...\n")


    snapshot_download(

        repo_id=repo_id,

        local_dir=path

    )


    metadata = get_model_metadata(
        repo_id
    )


    record = {

        "repo_id": repo_id,

        "organization": organization,

        "model_name": model_name,

        "location": path,

        "author": metadata["author"],

        "license": metadata["license"],

        "huggingface_downloads":
            metadata["downloads"],

        "download_date":
            str(datetime.now()),

        "status":
            "completed"
    }


    update_inventory(
        record
    )


    print("\n================================")
    print("Download Completed")
    print("================================")



if __name__ == "__main__":

    import sys


    if len(sys.argv) < 2:

        print(
            "\nUsage:"
            "\npython downloader.py organization/model_name"
            "\n\nExample:"
            "\npython downloader.py deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        )

    else:

        download_model(
            sys.argv[1]
        )