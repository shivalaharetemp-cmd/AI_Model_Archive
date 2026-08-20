from huggingface_hub import snapshot_download
import json
import os
from datetime import datetime


BASE_DIR = "models"


def select_folder(repo_id):

    name = repo_id.split("/")[1].lower()


    if "qwen" in name:
        return "Qwen"

    elif "deepseek" in name:
        return "DeepSeek"

    elif "llama" in name:
        return "Llama"

    elif "gemma" in name:
        return "Google"

    elif "mistral" in name:
        return "Mistral"

    else:
        return "Other"



def update_inventory(data):

    file = "inventory.json"


    if os.path.exists(file):

        with open(file,"r") as f:
            inventory=json.load(f)

    else:
        inventory=[]


    inventory.append(data)


    with open(file,"w") as f:

        json.dump(
            inventory,
            f,
            indent=4
        )




def download_model(repo_id):


    family = select_folder(repo_id)


    model_name = repo_id.split("/")[1]


    path = os.path.join(
        BASE_DIR,
        family,
        model_name
    )


    print("\nDownloading:")
    print(repo_id)

    print("\nSaving to:")
    print(path)



    confirm=input(
        "\nContinue? (yes/no): "
    )


    if confirm.lower() != "yes":

        print("Cancelled")
        return



    snapshot_download(

        repo_id=repo_id,

        local_dir=path

    )


    record={

        "model": repo_id,

        "location": path,

        "download_date":
            str(datetime.now()),

        "status":
            "completed"

    }


    update_inventory(record)


    print("\nDownload completed")




if __name__=="__main__":

    import sys


    if len(sys.argv)<2:

        print(
        "Usage:\n"
        "python downloader.py organization/model"
        )

    else:

        download_model(
            sys.argv[1]
        )