#!/usr/bin/env python3
"""
Download TRELLIS.2 model weights and dependencies.

Total download: ~20 GB
Cache directory: ~/.cache/trellis2/ (configurable via TRELLIS2_CACHE env var)

Gated models (need HuggingFace access):
  - DinoV3: legacy ModelScope mirror; original model license/access terms still apply
  - RMBG-2.0 (briaai/RMBG-2.0) → needs HF_TOKEN
    1. Visit https://huggingface.co/briaai/RMBG-2.0 → "Agree and access repository"
    2. Create token at https://huggingface.co/settings/tokens
    3. export HF_TOKEN=your_token
"""
import os, json, shutil

CACHE = os.environ.get("TRELLIS2_CACHE", os.path.expanduser("~/.cache/trellis2"))
PIPELINE_DIR = os.path.join(CACHE, "pipeline")
IMAGE_DIR = os.path.join(CACHE, "image")
DOWNLOADS = {}


def download_hf_snapshot(repo, destination, revision='main', token=None):
    """Resolve a mutable ref once and record the exact revision actually downloaded."""
    from huggingface_hub import HfApi, snapshot_download
    from pathlib import Path
    resolved = HfApi().model_info(repo, revision=revision, token=token).sha
    if not resolved:
        raise ValueError(f'No resolved checkpoint revision for {repo}')
    path = snapshot_download(repo_id=repo, revision=resolved, local_dir=destination, token=token)
    DOWNLOADS[repo] = {'revision': resolved, 'requested_revision': revision}
    Path(PIPELINE_DIR).mkdir(parents=True, exist_ok=True)
    (Path(PIPELINE_DIR) / 'download_manifest.json').write_text(
        json.dumps(DOWNLOADS, indent=2), encoding='utf-8')
    return path


def download_file(url, dest, desc=""):
    """Download with progress bar."""
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"  Downloading {desc}...")
    print(f"    {url}")

    def report(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
        mb = min(total_size, count * block_size) / 1024**2
        total_mb = total_size / 1024**2
        print(f"\r    {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()
    return dest


def download_trellis2_4b():
    """Download TRELLIS.2-4B model (from HuggingFace)."""
    print("\n[1/4] TRELLIS.2-4B (~16 GB)")

    repo = "microsoft/TRELLIS.2-4B"

    try:
        path = download_hf_snapshot(repo, PIPELINE_DIR, os.environ.get('TRELLIS2_MODEL_REVISION', 'main'))
        # Localized pipeline.json must never become the source of the next official backup.
        from huggingface_hub import hf_hub_download
        original = hf_hub_download(repo_id=repo, filename='pipeline.json', revision=DOWNLOADS[repo]['revision'])
        shutil.copyfile(original, os.path.join(PIPELINE_DIR, 'pipeline.json'))
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Try manual download:")
        print(f"    pip install huggingface_hub")
        print(f"    export HF_ENDPOINT=https://hf-mirror.com")
        print(f"    huggingface-cli download {repo} --local-dir {PIPELINE_DIR}")
        raise


def download_image_large():
    """Download TRELLIS-image-large (~3.1 GB)."""
    print("\n[2/4] TRELLIS-image-large (~3.1 GB)")

    repo = "microsoft/TRELLIS-image-large"

    try:
        path = download_hf_snapshot(repo, IMAGE_DIR, os.environ.get('TRELLIS_IMAGE_REVISION', 'main'))
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def download_dinov3():
    """Download the legacy DINOv3 mirror; original model terms still apply."""
    print("\n[3/4] DinoV3 ViT-L/16 (~1.2 GB)")

    try:
        from modelscope import snapshot_download
        path = snapshot_download(
            "tastelikefeet/dinov3-vitl16-pretrain-lvd1689m",
            cache_dir=os.path.join(CACHE, "modelscope"),
        )
        print(f"  OK: {path}")
        return path
    except ImportError:
        raise RuntimeError('ModelScope missing; install requirements.txt before downloading')


def download_rmbg2():
    """Download RMBG-2.0 (needs HF token for gated access)."""
    print("\n[4/4] RMBG-2.0 (~1 GB)")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("  WARNING: HF_TOKEN not set. RMBG-2.0 requires HuggingFace access.")
        print("  1. Visit https://huggingface.co/briaai/RMBG-2.0")
        print("  2. Click 'Agree and access repository'")
        print("  3. Create token at https://huggingface.co/settings/tokens")
        print("  4. Run: export HF_TOKEN=your_token")
        print("  Skipping RMBG-2.0 download for now...")
        return None

    try:
        path = download_hf_snapshot('briaai/RMBG-2.0', os.path.join(CACHE, 'rmbg2'), token=token)
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  RMBG-2.0 download failed. Background removal will be unavailable.")


def create_pipeline_config(dinov3_path):
    """Only localize paths; preserve the downloaded official numeric config."""
    import copy
    from pathlib import Path
    config_path = Path(PIPELINE_DIR) / "pipeline.json"
    backup_path = Path(PIPELINE_DIR) / "pipeline.official.json"
    original = json.loads(config_path.read_text(encoding="utf-8"))
    if original.get("name") != "Trellis2ImageTo3DPipeline":
        raise ValueError("Unexpected official pipeline configuration")
    # main() refreshes pipeline.json from the pinned snapshot before this call.
    backup_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    config = copy.deepcopy(original)
    args = config["args"]
    args["models"]["sparse_structure_decoder"] = os.path.abspath(os.path.join(
        IMAGE_DIR, "ckpts", "ss_dec_conv3d_16l8_fp16"))
    if dinov3_path:
        args["image_cond_model"]["args"]["model_name"] = os.path.abspath(dinov3_path)
    local_rembg = os.path.join(CACHE, "rmbg2")
    if os.path.isfile(os.path.join(local_rembg, "config.json")):
        args["rembg_model"]["args"]["model_name"] = os.path.abspath(local_rembg)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Config localized: {config_path}; original preserved: {backup_path}")



# ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRELLIS.2 Weight Downloader")
    print(f"  Cache: {CACHE}")
    print("=" * 60)

    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 1-2: Main model weights
    download_trellis2_4b()
    download_image_large()

    # 3: DinoV3
    dinov3_path = download_dinov3()

    # 4: RMBG-2.0
    download_rmbg2()

    # Create config
    create_pipeline_config(dinov3_path)

    print(f"\n{'=' * 60}")
    print("  Download complete!")
    print(f"  Cache: {CACHE}")
    print(f"  Run: python generate.py <image> -o output.glb")
    print("=" * 60)


if __name__ == "__main__":
    main()
