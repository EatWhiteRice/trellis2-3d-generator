import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('experiment')

    def test_unreadable_asset_is_not_perfect_color(self):
        # Stub background removal: this regression must never download a model.
        rembg = types.ModuleType('rembg')
        rembg.remove = lambda image: image.convert('RGBA')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'rembg': rembg}):
            image = Path(tmp) / 'input.png'
            Image.new('RGB', (12, 12), 'red').save(image)
            with self.assertRaisesRegex(ValueError, 'texture|asset|GLB'):
                self.module.compute_color_error(Path(tmp) / 'missing.glb', image)

    def test_unreadable_asset_has_no_fake_edge_fill(self):
        with self.assertRaises(ValueError):
            self.module.compute_edge_fill('missing.glb')


class EntryPointRegressionTests(unittest.TestCase):
    def test_mock_generation_preserves_alpha_raw_geometry_and_metadata(self):
        module = load_module('generate')
        class Tensor:
            def __init__(self, array): self.array = np.asarray(array); self.shape = self.array.shape
            def cpu(self): return self
            def detach(self): return self
            def numpy(self): return self.array
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        class Mesh:
            def __init__(self):
                self.vertices = Tensor(vertices.copy()); self.faces = Tensor(faces)
                self.attrs = Tensor(np.zeros((3, 6))); self.coords = Tensor(np.zeros((3, 3)))
                self.layout = {}; self.voxel_size = 1 / 512
            def simplify(self, target): self.vertices.array += 10
        calls = []
        pipeline = types.SimpleNamespace(cuda=lambda: None)
        pipeline.run = lambda image, **kw: calls.append((image.mode, kw)) or [Mesh()]
        pipelines = types.ModuleType('trellis2.pipelines')
        pipelines.Trellis2ImageTo3DPipeline = types.SimpleNamespace(from_pretrained=lambda cache: pipeline)
        torch = types.ModuleType('torch'); torch.__version__ = 'mock'; torch.version = types.SimpleNamespace(cuda='mock')
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'mock',
            mem_get_info=lambda: (100, 200), synchronize=lambda: None, empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 12)
        voxel = types.ModuleType('o_voxel')
        voxel.postprocess = types.SimpleNamespace(to_glb=lambda **kw: types.SimpleNamespace(
            export=lambda path, **kwargs: Path(path).write_bytes(b'mock-export-only')))
        replacements = {'torch': torch, 'trellis2': types.ModuleType('trellis2'), 'trellis2.pipelines': pipelines, 'o_voxel': voxel}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); image = tmp / 'input.png'; output = tmp / 'nested' / 'out.glb'
            Image.new('RGBA', (4, 4), (100, 80, 60, 0)).save(image)
            cache = tmp / 'pipeline'; cache.mkdir()
            (cache / 'pipeline.json').write_text(json.dumps({'name':'Trellis2ImageTo3DPipeline', 'args':{'models':{}}}))
            with patch.dict(sys.modules, replacements), patch.object(module, 'find_trellis_root', return_value=str(tmp)), patch.object(module, 'check_cache', return_value=str(cache)), patch.object(sys, 'argv', ['generate.py', str(image), '-o', str(output), '--seed', '17', '--save-raw']):
                module.main()
            with np.load(output.with_suffix('.raw_geometry.npz'), allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['vertices'], vertices)
            metadata = json.loads(output.with_suffix('.metadata.json').read_text())
            self.assertEqual(metadata['seed'], 17)
            self.assertEqual(metadata['raw_geometry_counts']['faces'], 1)
            self.assertEqual(metadata['gpu'], 'mock')
            self.assertIn('config_sha256', metadata)
        self.assertEqual(calls[0][0], 'RGBA')
        self.assertEqual(calls[0][1]['seed'], 17)

    def test_help_does_not_require_torch_or_trellis(self):
        # Explicitly block torch even if it happens to be installed on the host.
        code = "import sys,runpy; sys.modules['torch']=None; sys.argv=['generate.py','--help']; runpy.run_path('generate.py',run_name='__main__')"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('--seed', result.stdout)

    def test_no_pbr_is_rejected_instead_of_silently_ignored(self):
        code = "import sys,runpy; sys.modules['torch']=None; sys.argv=['generate.py','image.png','--no-pbr']; runpy.run_path('generate.py',run_name='__main__')"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertIn('unrecognized arguments', result.stderr)


class ConfigurationRegressionTests(unittest.TestCase):
    def test_local_paths_preserve_official_normalization_and_sampler(self):
        module = load_module('download_weights')
        original = {'name': 'Trellis2ImageTo3DPipeline', 'args': {
            'models': {'sparse_structure_decoder': 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'},
            'shape_slat_normalization': {'mean': [0.781296], 'std': [5.972266]},
            'tex_slat_normalization': {'mean': [3.501659], 'std': [2.665652]},
            'sparse_structure_sampler': {'params': {'guidance_strength': 7.5}},
            'image_cond_model': {'name': 'DinoV3FeatureExtractor', 'args': {'model_name': 'upstream/dino'}},
            'rembg_model': {'name': 'BiRefNet', 'args': {'model_name': 'briaai/RMBG-2.0'}},
            'default_pipeline_type': '1024_cascade'}}
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Path(tmp) / 'pipeline'; pipeline.mkdir()
            (pipeline / 'pipeline.json').write_text(json.dumps(original), encoding='utf-8')
            with patch.object(module, 'PIPELINE_DIR', str(pipeline)), patch.object(module, 'IMAGE_DIR', str(Path(tmp) / 'image')):
                module.create_pipeline_config('/local/dino')
                result = json.loads((pipeline / 'pipeline.json').read_text(encoding='utf-8'))
                backup = json.loads((pipeline / 'pipeline.official.json').read_text(encoding='utf-8'))
            self.assertEqual(backup, original)
            for key in ['shape_slat_normalization', 'tex_slat_normalization', 'sparse_structure_sampler', 'default_pipeline_type']:
                self.assertEqual(result['args'][key], original['args'][key])
            self.assertEqual(result['args']['image_cond_model']['args']['model_name'], str(Path('/local/dino').resolve()))

    def test_snapshot_download_uses_resolved_revision(self):
        module = load_module('download_weights')
        calls = []
        hub = types.ModuleType('huggingface_hub')
        hub.HfApi = lambda: types.SimpleNamespace(model_info=lambda *a, **kw: types.SimpleNamespace(sha='resolved-revision'))
        hub.snapshot_download = lambda **kw: calls.append(kw) or kw['local_dir']
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'huggingface_hub': hub}), patch.object(module, 'PIPELINE_DIR', tmp):
            module.download_hf_snapshot('official/model', tmp)
            manifest = json.loads((Path(tmp) / 'download_manifest.json').read_text())
        self.assertEqual(calls[0]['revision'], 'resolved-revision')
        self.assertEqual(manifest['official/model']['revision'], 'resolved-revision')


class GeometryRegressionTests(unittest.TestCase):
    def test_gltf_y_up_is_rotated_for_simulation_and_back_for_export(self):
        import trimesh
        from mesh_utils import load_normalized_mesh
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'tall.glb'
            trimesh.creation.box(extents=[1, 4, 1]).export(asset)
            mesh = load_normalized_mesh(asset)
            self.assertAlmostEqual(mesh.extents[2], 0.5)
            self.assertAlmostEqual(mesh.extents[1], 0.125)
            from mesh_utils import simulation_to_gltf
            exported = simulation_to_gltf(mesh)
            self.assertAlmostEqual(exported.extents[1], 0.5)
            self.assertAlmostEqual(exported.bounds[0, 1], 0)

    def test_box_collision_and_visual_frames_follow_mesh_bounds(self):
        import trimesh
        module = load_module('sim_pipeline')
        mesh = trimesh.creation.box(); mesh.apply_translation([0.2, 0.1, 0.5])
        expected_center = mesh.bounds.mean(axis=0)
        bullet = types.ModuleType('pybullet'); bullet.DIRECT = 0; bullet.GEOM_BOX = 3
        collision_calls = []; visual_calls = []
        bullet.createCollisionShape = lambda kind, **kw: collision_calls.append(kw) or 1
        bullet.createVisualShape = lambda kind, **kw: visual_calls.append(kw) or 2
        bullet.createMultiBody = lambda **kw: 3
        bullet.getBasePositionAndOrientation = lambda bid: ([0, 0, 1], [0, 0, 0, 1])
        bullet.getMatrixFromQuaternion = lambda q: np.eye(3).ravel().tolist()
        for name in ['connect', 'setGravity', 'setTimeStep', 'setAdditionalSearchPath', 'loadURDF', 'disconnect']:
            setattr(bullet, name, lambda *a: None)
        data = types.ModuleType('pybullet_data'); data.getDataPath = lambda: '/mock'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'pybullet': bullet, 'pybullet_data': data}):
            module.sim_drop([mesh], ['box'], output=str(Path(tmp) / 'out.glb'), duration=0)
        np.testing.assert_allclose(collision_calls[0]['collisionFramePosition'], expected_center)
        np.testing.assert_allclose(visual_calls[0]['visualFramePosition'], expected_center)

    def test_rigid_collision_shape_receives_explicit_hull_faces(self):
        import trimesh
        bullet = types.ModuleType('pybullet'); bullet.DIRECT = 0; bullet.GEOM_MESH = 5
        calls = []
        bullet.createCollisionShape = lambda kind, **kw: calls.append(kw) or 1
        bullet.createVisualShape = lambda *a, **kw: 2
        bullet.createMultiBody = lambda **kw: 3
        bullet.getBasePositionAndOrientation = lambda bid: ([0, 0, 1], [0, 0, 0, 1])
        bullet.getMatrixFromQuaternion = lambda q: np.eye(3).ravel().tolist()
        for name in ['connect', 'setGravity', 'setTimeStep', 'setAdditionalSearchPath', 'loadURDF', 'disconnect']:
            setattr(bullet, name, lambda *a: None)
        data = types.ModuleType('pybullet_data'); data.getDataPath = lambda: '/mock'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'pybullet': bullet, 'pybullet_data': data}):
            load_module('physics_sim').run_simulation([trimesh.creation.box()], ['box'],
                output_path=str(Path(tmp) / 'out.glb'), duration=0)
        self.assertIn('indices', calls[0])
        self.assertGreater(len(calls[0]['indices']), 0)

    def test_simulation_prep_uses_z_up_and_all_scene_parts(self):
        import trimesh
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'parts.glb'
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.box(), node_name='first')
            transform = np.eye(4); transform[0, 3] = 3
            scene.add_geometry(trimesh.creation.box(), node_name='second', transform=transform)
            scene.export(asset)
            for name in ['physics_sim', 'sim_pipeline']:
                mesh = load_module(name).prep_mesh(asset)
                self.assertEqual(len(mesh.vertices), 16)
                self.assertAlmostEqual(mesh.bounds[0, 2], 0)
                self.assertAlmostEqual(mesh.extents.max(), 0.5)

    def test_color_strength_zero_preserves_texture_and_pbr(self):
        import trimesh
        from trimesh.visual.material import PBRMaterial
        mesh = trimesh.creation.box()
        texture = Image.new('RGB', (8, 8), (100, 80, 60))
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)),
            material=PBRMaterial(baseColorTexture=texture, metallicFactor=0.8, roughnessFactor=0.2))
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'textured.glb'; mesh.export(asset)
            stats = {'mean': np.array([130, 140, 150]), 'std': np.array([5, 5, 5])}
            scene = load_module('post_process').calibrate_colors(asset, stats, strength=0)
            material = next(iter(scene.geometry.values())).visual.material
            np.testing.assert_array_equal(np.asarray(material.baseColorTexture.convert('RGB')), np.asarray(texture))
            self.assertAlmostEqual(material.metallicFactor, 0.8)
            self.assertAlmostEqual(material.roughnessFactor, 0.2)


if __name__ == '__main__':
    unittest.main()
