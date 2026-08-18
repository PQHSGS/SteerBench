"""Single-file CLI to upload GLP and subspace GLP checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

from huggingface_hub import HfApi, login
import logging


@dataclass(frozen=True)
class UploadFile:
    source_name: str
    target_name: str


@dataclass(frozen=True)
class UploadSpec:
    name: str
    required_source_files: tuple[str, ...]
    files: tuple[UploadFile, ...]


GLP_UPLOAD_SPEC = UploadSpec(
    name="glp",
    required_source_files=("final.safetensors", "rep_statistics.pt", "config.yaml"),
    files=(
        UploadFile("final.safetensors", "final.safetensors"),
        UploadFile("rep_statistics.pt", "rep_statistics.pt"),
        UploadFile("config.yaml", "config.yaml"),
        # subspace artifacts not included for standard GLP checkpoints
    ),
)


SUBSPACE_GLP_UPLOAD_SPEC = UploadSpec(
    name="subspace",
    required_source_files=(
        "final.safetensors",
        "config.yaml",
        "rep_statistics.pt",
        "subspace.pt",
        "mean_P.pt",
    ),
    files=(
        UploadFile("final.safetensors", "final.safetensors"),
        UploadFile("config.yaml", "config.yaml"),
        UploadFile("rep_statistics.pt", "rep_statistics.pt"),
        UploadFile("subspace.pt", "subspace.pt"),
        UploadFile("mean_P.pt", "mean_P.pt"),
    ),
)


def ensure_login(token: str | None) -> None:
    if token:
        login(token=token)


def normalize_repo_path(path_in_repo: str) -> str:
    normalized = str(path_in_repo).replace("\\", "/").strip()
    normalized = normalized.strip("/")
    if normalized in {"", "."}:
        raise ValueError("path_in_repo resolves to repository root; provide a non-empty subfolder")
    return normalized


def target_path_exists(existing_files: set[str], target_path: str) -> bool:
    prefix = f"{target_path}/"
    return any(path == target_path or path.startswith(prefix) for path in existing_files)


def validate_required_files(folder_path: Path, required_files: Sequence[str]) -> list[str]:
    # Keep for backward compatibility but don't enforce strict failure.
    return [file_name for file_name in required_files if not (folder_path / file_name).exists()]


def upload_checkpoint_folder(
    *,
    api: HfApi,
    repo_id: str,
    repo_type: str,
    folder_path: Path,
    target_path: str,
    spec: UploadSpec,
    allow_overlap: bool,
    upload_all: bool = False,
) -> None:
    existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    if target_path_exists(existing_files, target_path) and not allow_overlap:
        raise ValueError(
            f"target path '{target_path}' already exists in {repo_id}; use --allow-overlap only if overwriting is intended"
        )

    logger = logging.getLogger(__name__)

    def _find_source(folder: Path, name: str) -> Optional[Path]:
        # Try exact name
        cand = folder / name
        if cand.exists():
            return cand
        # try variants with/without _final
        if name.endswith("_final.yaml") or name.endswith("_final.pt"):
            alt = folder / name.replace("_final", "")
            if alt.exists():
                return alt
        else:
            alt = folder / (Path(name).stem + "_final" + Path(name).suffix)
            if alt.exists():
                return alt
        # special cases
        if name == "config.yaml":
            for candname in ("config_final.yaml", "config.yaml"):
                if (folder / candname).exists():
                    return folder / candname
        if name == "rep_statistics.pt":
            for candname in ("rep_statistics.pt", "rep_statistics_final.pt"):
                if (folder / candname).exists():
                    return folder / candname
        if name == "final.safetensors":
            for candname in ("final.safetensors",):
                if (folder / candname).exists():
                    return folder / candname
        return None

    if upload_all:
        # Upload every file under folder_path recursively, preserving relative paths.
        for src in sorted(folder_path.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(folder_path).as_posix()
            repo_path = f"{target_path}/{rel}"
            logger.info("Uploading %s -> %s", src, repo_path)
            api.upload_file(path_or_fileobj=str(src), path_in_repo=repo_path, repo_id=repo_id, repo_type=repo_type)
        return

    for upload_file in spec.files:
        source_path = _find_source(folder_path, upload_file.source_name)
        if source_path is None:
            logger.warning("Missing optional upload source '%s' in %s — skipping.", upload_file.source_name, folder_path)
            continue

        repo_path = f"{target_path}/{upload_file.target_name}"
        logger.info("Uploading %s -> %s", source_path, repo_path)
        api.upload_file(
            path_or_fileobj=str(source_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push GLP checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo ID like 'username/my-model'")
    parser.add_argument("--token", default=None, help="HF Token (optional, otherwise reads from HF_TOKEN or credentials)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Target subfolder in the HF repo. Defaults to the local folder name.",
    )
    parser.add_argument(
        "--allow-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow uploading to an existing target path in the repo.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    glp_parser = subparsers.add_parser("glp", help="Upload a standard GLP training checkpoint")
    glp_parser.add_argument("--folder", required=True, help="Folder with final.safetensors, rep_statistics.pt, config.yaml")

    # Subspace upload command removed; only standard GLP checkpoints are supported.

    return parser


def _validate_folder(folder_path: Path, spec_name: str, required_files: tuple[str, ...]) -> bool:
    missing_files = validate_required_files(folder_path, required_files)
    if not missing_files:
        return False

    print(f"Warning: the {spec_name} folder is missing required files:")
    for file_name in missing_files:
        print(f"  - {file_name}")

    # Ask user whether to continue uploading the available files (upload entire folder).
    try:
        # Interactive prompt — default is No.
        answer = input("Upload entire folder contents instead? [y/N]: ").strip().lower()
    except Exception:
        # If input is unavailable (non-interactive), abort to be safe.
        print("Non-interactive environment: aborting upload due to missing files.")
        sys.exit(1)

    if answer in ("y", "yes"):
        print("Proceeding with upload of all files found in the folder...")
        return True

    print("Aborting upload.")
    sys.exit(1)


def _upload(args, spec) -> None:
    ensure_login(args.token)

    api = HfApi()
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{args.folder}' does not exist or is not a directory.")
        sys.exit(1)

    target_path = args.path_in_repo or folder_path.name
    try:
        target_path = normalize_repo_path(target_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    upload_all = _validate_folder(folder_path, spec.name, spec.required_source_files)

    print(f"Creating repo {args.repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)

    print(f"Uploading {spec.name} checkpoint from {folder_path} to {args.repo_id}:{target_path}...")
    try:
        upload_checkpoint_folder(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=folder_path,
            target_path=target_path,
            spec=spec,
            allow_overlap=args.allow_overlap,
            upload_all=bool(upload_all),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Upload complete!")


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "glp":
        _upload(args, GLP_UPLOAD_SPEC)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()