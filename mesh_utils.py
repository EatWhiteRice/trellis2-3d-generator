"""CPU mesh loading for the experimental Z-up rigid-body demos."""
import numpy as np
import trimesh


def load_normalized_mesh(path, target_size=0.5, input_up='y'):
    if not np.isfinite(target_size) or target_size <= 0:
        raise ValueError('target_size must be finite and positive')
    scene = trimesh.load(path, force='scene')
    # Applies scene-graph transforms and includes every mesh part, not just the first.
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError('Asset contains no triangle mesh')
    if not np.isfinite(mesh.vertices).all() or mesh.extents.max() <= 0:
        raise ValueError('Asset has invalid or degenerate coordinates')
    if input_up == 'y':
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif input_up != 'z':
        raise ValueError('input_up must be y or z')
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices *= target_size / mesh.extents.max()
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
    return mesh


def simulation_to_gltf(mesh):
    """Export standard Y-up glTF from a Z-up simulation mesh without mutating it."""
    result = mesh.copy()
    result.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return result
