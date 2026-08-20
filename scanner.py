from huggingface_hub import HfApi
import sys


def bytes_to_gb(size):
    return size / (1024 ** 3)


def scan_model(repo_id):

    api = HfApi()

    print("\n================================")
    print("MODEL SCANNER")
    print("================================\n")

    print(f"Model: {repo_id}\n")

    try:
        info = api.model_info(repo_id)

        print("Author:")
        print(info.author)

        print("\nLicense:")
        print(info.cardData.get("license", "Unknown")
              if info.cardData else "Unknown")


        print("\nFiles:")

        files = api.list_repo_tree(
            repo_id,
            recursive=True
        )


        total_size = 0
        count = 0


        for file in files:

            if hasattr(file, "size") and file.size:

                total_size += file.size
                count += 1

                print(
                    f"{file.path:70}"
                    f"{bytes_to_gb(file.size):8.2f} GB"
                )


        print("\n================================")
        print(f"Total files : {count}")
        print(
            f"Total size  : {bytes_to_gb(total_size):.2f} GB"
        )
        print("================================")


    except Exception as e:

        print("\nERROR:")
        print(e)



if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python scanner.py organization/model"
        )

    else:

        scan_model(sys.argv[1])