#!/usr/bin/env python3
"""Inspect a GLB/glTF file and print scene statistics.

Usage:
  python tools/inspect_glb.py --path /path/to/scene_download.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_shape(obj: Any) -> str:
    shape = getattr(obj, "shape", None)
    if shape is None:
        return "n/a"
    return "x".join(str(int(x)) for x in shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GLB content (meshes, point clouds, cameras, nodes)")
    parser.add_argument("--path", required=True, help="Path to .glb/.gltf file")
    parser.add_argument("--dump-nodes", action="store_true", help="Print node transform info")
    parser.add_argument("--dump-materials", action="store_true", help="Print material names and count")
    args = parser.parse_args()

    glb_path = Path(args.path)
    if not glb_path.exists():
        raise FileNotFoundError(f"File not found: {glb_path}")

    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Missing dependency 'trimesh'. Install with: pip install trimesh") from exc

    scene_or_mesh = trimesh.load(str(glb_path), force="scene")
    scene = scene_or_mesh if hasattr(scene_or_mesh, "geometry") else trimesh.Scene(scene_or_mesh)

    print(f"file: {glb_path}")
    print(f"geometries: {len(scene.geometry)}")
    print(f"graph nodes: {len(scene.graph.nodes_geometry)}")

    mesh_count = 0
    pointcloud_count = 0
    camera_like = 0

    for name, geom in scene.geometry.items():
        gtype = type(geom).__name__
        verts = getattr(geom, "vertices", None)
        pts = getattr(geom, "vertices", None)
        faces = getattr(geom, "faces", None)
        colors = getattr(geom, "colors", None)

        if gtype.lower().endswith("pointcloud"):
            pointcloud_count += 1
            point_n = 0 if pts is None else int(len(pts))
            print(f"- geometry[{name}] type=PointCloud points={point_n} colors_shape={_safe_shape(colors)}")
        else:
            mesh_count += 1
            vert_n = 0 if verts is None else int(len(verts))
            face_n = 0 if faces is None else int(len(faces))
            print(f"- geometry[{name}] type={gtype} vertices={vert_n} faces={face_n} colors_shape={_safe_shape(colors)}")

    # Heuristic: DUSt3R exports camera frustums as extra mesh geometry.
    for node_name in scene.graph.nodes_geometry:
        if "camera" in str(node_name).lower() or "cam" in str(node_name).lower():
            camera_like += 1

    print(f"mesh geometries: {mesh_count}")
    print(f"pointcloud geometries: {pointcloud_count}")
    print(f"camera-like nodes (name heuristic): {camera_like}")

    # Global bounds.
    try:
        b = scene.bounds
        print(f"bounds_min: {b[0].tolist()}")
        print(f"bounds_max: {b[1].tolist()}")
        ext = (b[1] - b[0]).tolist()
        print(f"extent: {ext}")
    except Exception:
        print("bounds: unavailable")

    if args.dump_nodes:
        print("nodes:")
        for node_name in scene.graph.nodes_geometry:
            try:
                transform, geom_name = scene.graph[node_name]
                row0 = [float(x) for x in transform[0].tolist()]
                t = [float(x) for x in transform[:3, 3].tolist()]
                payload = {
                    "node": str(node_name),
                    "geometry": str(geom_name),
                    "translation": t,
                    "m00_m01_m02_m03": row0,
                }
                print(json.dumps(payload, ensure_ascii=True))
            except Exception:
                print(json.dumps({"node": str(node_name), "error": "cannot_read_transform"}, ensure_ascii=True))

    if args.dump_materials:
        mats = []
        for _, geom in scene.geometry.items():
            visual = getattr(geom, "visual", None)
            material = getattr(visual, "material", None) if visual is not None else None
            mats.append(getattr(material, "name", None))
        print(f"materials_count: {len([m for m in mats if m is not None])}")
        print(f"materials: {mats}")


if __name__ == "__main__":
    main()
