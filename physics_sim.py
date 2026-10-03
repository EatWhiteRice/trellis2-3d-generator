#!/usr/bin/env python3
"""
Offline Physics Simulation Pipeline
GLB → Z-up normalization → single convex hull → rigid PyBullet drop → result

Usage:
  python physics_sim.py output_apple.glb           # single object drop
  python physics_sim.py A.glb B.glb                 # multi-object collision
  python physics_sim.py output_apple.glb --stack 3  # stack N copies
"""
import os, sys, time
import numpy as np
import trimesh


# ═══════════════════════════════════════════════════════════
# STEP 1: Mesh Prep — normalize, center, scale
# ═══════════════════════════════════════════════════════════

def prep_mesh(glb_path, target_size=0.5, input_up='y'):
    from mesh_utils import load_normalized_mesh
    return load_normalized_mesh(glb_path, target_size, input_up)



# ═══════════════════════════════════════════════════════════
# Collision uses a single convex hull inside run_simulation.
# ═══════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════
# STEP 3: PyBullet Simulation
# ═══════════════════════════════════════════════════════════

def run_simulation(meshes, labels, output_path="sim_result.glb", duration=3.0):
    """
    Run PyBullet physics simulation:
    - Drop objects from height
    - Let them collide and settle
    - Export final poses as GLB scene

    meshes: list of trimesh.Trimesh (already prepped)
    labels: list of names
    """
    import pybullet as p
    import pybullet_data

    # Connect
    client = p.connect(p.DIRECT)  # headless
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Ground plane
    p.loadURDF("plane.urdf")

    body_ids = []
    initial_heights = []

    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        # A single convex hull is an approximation; this is not articulated simulation.
        hull = mesh.convex_hull
        col_id = p.createCollisionShape(
            p.GEOM_MESH, vertices=hull.vertices.tolist(),
            indices=hull.faces.reshape(-1).tolist(), meshScale=[1, 1, 1])

        # Visual shape (same mesh)
        vis_id = p.createVisualShape(
            p.GEOM_MESH, vertices=mesh.vertices.tolist(),
            indices=mesh.faces.reshape(-1).tolist(), meshScale=[1, 1, 1],
            rgbaColor=[0.8, 0.2, 0.1, 1.0])

        # Drop from height, spread out
        drop_z = 1.5 + i * 0.8
        pos = [i * 0.6 - (len(meshes)-1)*0.3, 0, drop_z]

        body_id = p.createMultiBody(
            baseMass=0.15,  # ~apple mass in kg
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
        )
        body_ids.append(body_id)
        initial_heights.append(drop_z)
        print(f"  [{label}] body_id={body_id}, drop_z={drop_z:.1f}m, mass=0.15kg")

    # Simulate
    fps = 240
    total_steps = int(duration * fps)
    print(f"\n  Simulating {duration}s at {fps}Hz ({total_steps} steps)...")

    start_time = time.time()
    for step in range(total_steps):
        p.stepSimulation()

        if step % (fps // 4) == 0:
            t = step / fps
            heights = []
            for bid in body_ids:
                pos, _ = p.getBasePositionAndOrientation(bid)
                heights.append(pos[2])
            print(f"    t={t:.1f}s  heights={[f'{h:.3f}' for h in heights]}")

    sim_time = time.time() - start_time
    print(f"  Done in {sim_time:.1f}s")

    # Get final poses
    final_meshes = []
    for i, (bid, mesh) in enumerate(zip(body_ids, meshes)):
        pos, orn = p.getBasePositionAndOrientation(bid)
        # Transform mesh to final pose
        T = np.eye(4)
        T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        T[:3, 3] = pos
        transformed = mesh.copy()
        transformed.apply_transform(T)
        from mesh_utils import simulation_to_gltf
        final_meshes.append(simulation_to_gltf(transformed))
        print(f"  [{labels[i]}] final pos={[f'{v:.3f}' for v in pos]}")

    p.disconnect()

    # Export as combined scene
    combined = trimesh.Scene()
    for i, (fm, label) in enumerate(zip(final_meshes, labels)):
        combined.add_geometry(fm, node_name=label)
    combined.export(output_path)
    print(f"\n  Exported: {output_path} ({os.path.getsize(output_path)/1024**2:.1f} MB)")

    return final_meshes


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline Physics Simulation Pipeline")
    parser.add_argument("glb_files", nargs="+", help="Input GLB file(s)")
    parser.add_argument("--stack", type=int, default=0, help="Stack N copies of first GLB")
    parser.add_argument("--output", default="sim_result.glb")
    parser.add_argument("--duration", type=float, default=3.0, help="Sim duration in seconds")
    parser.add_argument("--size", type=float, default=0.5, help="Target object size in meters")
    parser.add_argument('--input-up', choices=['y', 'z'], default='y', help='GLB convention is Y-up; override nonstandard assets')
    args = parser.parse_args()

    # Expand with --stack
    glb_list = list(args.glb_files)
    if args.stack > 1:
        glb_list = [args.glb_files[0]] * args.stack

    print("=" * 60)
    print("  Physics Simulation Pipeline")
    print(f"  Objects: {len(glb_list)}")
    print("=" * 60)

    # Step 1: Prep
    print("\n[1/3] Mesh Preparation...")
    meshes = []
    labels = []
    for g in glb_list:
        label = os.path.splitext(os.path.basename(g))[0]
        print(f"  {g}")
        mesh = prep_mesh(g, target_size=args.size, input_up=args.input_up)
        meshes.append(mesh)
        labels.append(label)

    # Step 2: Run simulation
    print("\n[2/3] Physics Simulation...")
    run_simulation(meshes, labels, args.output, args.duration)

    # Step 3: Done
    print(f"\n[3/3] Complete: {args.output}")


if __name__ == "__main__":
    main()
