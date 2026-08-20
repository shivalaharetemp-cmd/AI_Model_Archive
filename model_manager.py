#!/usr/bin/env python3
"""
model_manager.py — single-file HF model archive manager

Replaces scanner.py + downloader.py.

Folder layout:
    AI_Model_Archive/
    ├── model_manager.py
    ├── inventory.json
    └── models/
        └── <organization>/
            └── <model_name>/

Auth:
    Looks for a Hugging Face token in this order:
      1. --token CLI flag (explicit override, if passed)
      2. HF_TOKEN in a .env file next to this script
      3. HF_TOKEN environment variable
      4. HUGGING_FACE_HUB_TOKEN environment variable
      5. Token cached locally via `huggingface-cli login`
    If none of those are found, falls back to anonymous (unauthenticated)
    access — this works fine for public repos, but gated/private repos
    will fail with a clear error telling you to authenticate.

    To use a .env file, create one next to model_manager.py:
        AI_Model_Archive/.env
            HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Usage:
    python model_manager.py Qwen/Qwen3-8B
    python model_manager.py deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    python model_manager.py Qwen/Qwen3-8B --yes      # skip confirmation
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, GatedRepoError
except ImportError:
    sys.exit(
        "Missing dependency. Install it with:\n"
        "    pip install huggingface_hub\n"
    )

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INVENTORY_PATH = BASE_DIR / "inventory.json"
ENV_PATH = BASE_DIR / ".env"


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def load_dotenv_token(var_name: str = "HF_TOKEN") -> str | None:
    """
    Minimal .env parser (no external dependency). Looks for `<var_name>=value`
    in a .env file next to this script. Ignores blank lines and comments.
    Strips surrounding quotes if present.
    """
    if not ENV_PATH.exists():
        return None
    try:
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == var_name:
                value = value.strip().strip('"').strip("'")
                return value or None
    except OSError:
        return None
    return None


def resolve_token(cli_token: str | None) -> str | None:
    """
    Return a usable HF token, or None to fall back to anonymous access.
    Order: --token flag > .env file > HF_TOKEN env > HUGGING_FACE_HUB_TOKEN env
           > cached login (huggingface-cli login) > anonymous.
    """
    if cli_token:
        print("Using token passed via --token.")
        return cli_token

    dotenv_token = load_dotenv_token("HF_TOKEN")
    if dotenv_token:
        print(f"Using HF_TOKEN from {ENV_PATH}.")
        return dotenv_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        print("Using token from environment variable.")
        return env_token

    try:
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            print("Using cached token from `huggingface-cli login`.")
            return cached
    except Exception:
        pass

    print("No HF token found — proceeding anonymously (public repos only).")
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_repo_parts(repo_id: str) -> tuple[str, str]:
    if "/" not in repo_id:
        sys.exit(f"Invalid repo_id '{repo_id}'. Expected format: organization/model_name")
    organization, model_name = repo_id.split("/", 1)
    return organization, model_name


def fetch_model_info(api: HfApi, repo_id: str) -> dict:
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except GatedRepoError:
        sys.exit(
            f"'{repo_id}' is a gated repo. You need an authenticated token with "
            f"access granted (set HF_TOKEN or use --token)."
        )
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            sys.exit(
                f"'{repo_id}' requires authentication. Set HF_TOKEN, run "
                f"`huggingface-cli login`, or pass --token."
            )
        sys.exit(f"Failed to fetch info for '{repo_id}': {e}")

    total_bytes = sum((f.size or 0) for f in (info.siblings or []) if f.size)
    return {
        "size_gb": round(total_bytes / (1024 ** 3), 2),
        "license": (info.card_data or {}).get("license", "unknown") if info.card_data else "unknown",
        "downloads": getattr(info, "downloads", 0) or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "gated": bool(getattr(info, "gated", False)),
        "num_files": len(info.siblings or []),
    }


def load_inventory() -> list:
    if INVENTORY_PATH.exists():
        try:
            return json.loads(INVENTORY_PATH.read_text())
        except json.JSONDecodeError:
            print(f"Warning: {INVENTORY_PATH} is corrupted, starting a fresh inventory.")
    return []


def update_inventory(record: dict) -> None:
    inventory = load_inventory()
    inventory = [r for r in inventory if r["repo_id"] != record["repo_id"]]
    inventory.append(record)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2))
    print(f"Inventory updated: {INVENTORY_PATH}")


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def download_model(repo_id: str, cli_token: str | None, skip_confirm: bool) -> None:
    token = resolve_token(cli_token)
    api = HfApi(token=token)

    organization, model_name = get_repo_parts(repo_id)
    path = MODELS_DIR / organization / model_name
    path.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching metadata for '{repo_id}'...")
    info = fetch_model_info(api, repo_id)

    print("\n--- Model details ---")
    print(f"Repo ID:      {repo_id}")
    print(f"Organization: {organization}")
    print(f"Model name:   {model_name}")
    print(f"Size:         {info['size_gb']} GB across {info['num_files']} files")
    print(f"License:      {info['license']}")
    print(f"Downloads:    {info['downloads']:,}")
    print(f"Likes:        {info['likes']:,}")
    print(f"Gated:        {info['gated']}")
    print(f"Save to:      {path}")
    print("----------------------\n")

    if not skip_confirm:
        confirm = input("Proceed with download? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    print(f"Downloading to: {path}")
    snapshot_download(repo_id=repo_id, local_dir=str(path), token=token)

    record = {
        "repo_id": repo_id,
        "organization": organization,
        "model_name": model_name,
        "size_gb": info["size_gb"],
        "license": info["license"],
        "downloads": info["downloads"],
        "location": str(path),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "auth": "token" if token else "anonymous",
    }
    update_inventory(record)
    print("\nDownload completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and archive a Hugging Face model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python model_manager.py Qwen/Qwen3-8B\n"
            "  python model_manager.py deepseek-ai/DeepSeek-R1-0528-Qwen3-8B\n"
            "  python model_manager.py Qwen/Qwen3-8B --yes\n"
            "\n"
            "Auth: put HF_TOKEN=hf_xxx in a .env file next to this script.\n"
            "If no token is found anywhere, downloads run anonymously.\n"
        ),
    )
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. Qwen/Qwen3-8B")
    parser.add_argument(
        "--token", default=None,
        help="Optional manual override. Normally not needed — put HF_TOKEN in .env instead.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_model(args.repo_id, args.token, args.yes)


if __name__ == "__main__":
    main()
