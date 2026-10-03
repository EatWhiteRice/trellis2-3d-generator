#!/usr/bin/env python3
"""
TRELLIS.2 Image-to-3D Generator
Usage: python generate.py <image> [-o output.glb] [-r 512|1024] [--steps 12]
"""
import argparse
import os, sys, time, gc, json
import numpy as np
from PIL import Image
from pathlib import Path
import hashlib
import subprocess

# ── Path setup ────────────────────────────────────────────
# TRELLIS.2 must be installed. Check common locations.
TRELLIS_PATHS = [
    os.path.expanduser("~/TRELLIS.2"),
    os.path.join(os.path.dirname(__file__), "TRELLIS.2"),
    "/root/TRELLIS.2",
]
def find_trellis_root(explicit=None):
    requested = explicit or os.environ.get('TRELLIS2_ROOT')
    candidates = [requested] if requested else TRELLIS_PATHS
    for candidate in candidates:
        root = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(os.path.join(root, 'trellis2', '__init__.py')):
            return root
    raise FileNotFoundError('Official TRELLIS.2 checkout missing; follow README or use --trellis-root')


def git_metadata(root):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git('status', '--porcelain')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': None if status is None else bool(status)}

# ── Cache paths ────────────────────────────────────────────
CACHE_BASE = os.environ.get("TRELLIS2_CACHE", os.path.expanduser("~/.cache/trellis2"))
PIPELINE_CACHE = os.path.join(CACHE_BASE, "pipeline")
IMAGE_CACHE = os.path.join(CACHE_BASE, "image")


def check_cache():
    """Verify model caches exist."""
    pipeline_json = os.path.join(PIPELINE_CACHE, "pipeline.json")
    if not os.path.exists(pipeline_json):
        print("ERROR: Model weights not found. Run download_weights.py first.")
        print(f"  Expected: {pipeline_json}")
        sys.exit(1)
    config = json.loads(Path(pipeline_json).read_text(encoding='utf-8'))
    if config.get('name') != 'Trellis2ImageTo3DPipeline':
        raise ValueError('Unexpected pipeline configuration')
    for model in config['args']['models'].values():
        prefix = Path(model)
        if model.startswith('ckpts/'):
            prefix = Path(PIPELINE_CACHE) / prefix
        elif not prefix.is_absolute():
            continue  # Upstream remote references may still require network access.
        for suffix in ['.json', '.safetensors']:
            if not Path(str(prefix) + suffix).is_file():
                raise FileNotFoundError(f'Checkpoint incomplete: {prefix}{suffix}')
    return PIPELINE_CACHE




def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Image-to-3D")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", default="output.glb", help="Output GLB path")
    parser.add_argument("-r", "--resolution", type=int, default=512, choices=[512, 1024],
                        help="Pipeline resolution (512 = lower VRAM)")
    parser.add_argument("--steps", type=int, default=12, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--trellis-root', help='Official checkout; also configurable with TRELLIS2_ROOT')
    parser.add_argument('--save-raw', action='store_true', help='Save vertices/faces before simplify/remesh as NPZ')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    # Check input
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    TRELLIS_ROOT = find_trellis_root(args.trellis_root)
    sys.path.insert(0, TRELLIS_ROOT)
    # Check cache
    cache = check_cache()

    # Environment
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for real TRELLIS.2 inference')
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != '.glb':
        parser.error('--output must have a .glb extension')
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.mem_get_info()
    print(f"VRAM: free={mem[0]/1e9:.1f}GB / total={mem[1]/1e9:.1f}GB")
    print(f"Input: {args.image}")
    print(f"Pipeline: {args.resolution}, steps={args.steps}")
    print()

    # ── Load pipeline ──────────────────────────────────
    print("[1/3] Loading pipeline...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cache)
    pipeline.low_vram = True
    pipeline.default_pipeline_type = str(args.resolution)
    pipeline.cuda()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0
    print(f"  Done in {load_seconds:.1f}s")

    # ── Load envmap ────────────────────────────────────
    # Export does not use an environment map; lighting belongs to later rendering.

    # ── Load image ─────────────────────────────────────
    print("[2/3] Processing image...")
    with Image.open(args.image) as original:
        image = original.convert('RGBA' if 'A' in original.getbands() or 'transparency' in original.info else 'RGB')
    print(f"  Size: {image.size}")

    # ── Inference ──────────────────────────────────────
    print("[3/3] Running inference...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    pipeline_type = str(args.resolution)
    mesh = pipeline.run(
        image,
        pipeline_type=pipeline_type,
        num_samples=1,
        seed=args.seed,
        sparse_structure_sampler_params={'steps': args.steps},
        shape_slat_sampler_params={'steps': args.steps},
        tex_slat_sampler_params={'steps': args.steps},
    )[0]

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    v = mesh.vertices.cpu().numpy()
    print(f"  Done in {dt:.1f}s")
    print(f"  Vertices: {v.shape[0]:,}  Faces: {mesh.faces.shape[0]:,}")
    print(f"  PBR channels: {mesh.attrs.shape[1]}")
    raw_counts = {'vertices': int(mesh.vertices.shape[0]), 'faces': int(mesh.faces.shape[0])}
    raw_path = output.with_suffix('.raw_geometry.npz') if args.save_raw else None
    if raw_path:
        np.savez_compressed(raw_path, vertices=v, faces=mesh.faces.detach().cpu().numpy())

    # ── Export ─────────────────────────────────────────
    print(f"\nExporting to {args.output}...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    mesh.simplify(5000000)
    torch.cuda.empty_cache()

    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000,
        texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0,
        verbose=False,
    )
    glb.export(args.output, extension_webp=True)
    sz = os.path.getsize(args.output) / 1024**2
    torch.cuda.synchronize()
    export_seconds = time.perf_counter() - t0
    metadata = {
        'kind': 'real_inference', 'seed': args.seed, 'pipeline_type': pipeline_type,
        'steps_per_stage': args.steps, 'input_sha256': hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((Path(cache) / 'pipeline.json').read_bytes()).hexdigest(),
        'samplers_from_config': {key: value for key, value in json.loads((Path(cache) / 'pipeline.json').read_text(encoding='utf-8'))['args'].items() if key.endswith('_sampler')},
        'trellis': git_metadata(TRELLIS_ROOT), 'wrapper': git_metadata(Path(__file__).resolve().parent),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'low_vram': True,
        'load_seconds': load_seconds, 'generation_seconds': dt, 'export_seconds': export_seconds,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'raw_geometry_counts': raw_counts, 'raw_geometry_dump': str(raw_path) if raw_path else None,
        'export': {'remesh': True, 'decimation_target': 500000, 'texture_size': 2048},
    }
    provenance = Path(cache) / 'download_manifest.json'
    metadata['downloads'] = json.loads(provenance.read_text(encoding='utf-8')) if provenance.is_file() else None
    metadata_path = output.with_suffix('.metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"  Done in {export_seconds:.1f}s → {args.output} ({sz:.1f} MB)")
    print(f'Metadata: {metadata_path}')


if __name__ == "__main__":
    main()
