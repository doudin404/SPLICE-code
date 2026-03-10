from typing import Optional, Tuple
from datetime import datetime

import torch
import trimesh
import os
import numpy as np

import options
# from model import splice_model
from model_onehalf import SPLICE_model
from snowman_logger import create_tagged_logger
logger = create_tagged_logger("MODEL_UTILS")

def load_model(opt: options.Options):
    return []#splice_model.SPLICE_Module.load_from_checkpoint(opt.splice_model_path, map_location=opt.device), opt

def load_model_onehalf(opt: options.Options) -> Tuple[SPLICE_model.SPLICE_Module, options.Options]:
    return SPLICE_model.SPLICE_Module.load_from_checkpoint(opt.splice_model_path, map_location=opt.device), opt

def export_mesh(mesh: Tuple[torch.Tensor, torch.Tensor], path: str = "./output/mesh.ply"):
    verts, faces = mesh
    
    # Convert to numpy arrays
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    # Ensure export directory exists
    export_dir = os.path.dirname(path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    
    # Create Trimesh object and export
    tri = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    tri.export(path)
    logger.info(f"Mesh exported to {path}")


def export_spheres(sphere_xyz: torch.Tensor, sphere_r: torch.Tensor, path: str = "./output/spheres.ply"):
    sphere_xyz_np = sphere_xyz.detach().cpu().numpy()
    sphere_r_np = sphere_r.detach().cpu().numpy()

    # Ensure export directory exists
    export_dir = os.path.dirname(path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    # Create sphere meshes and aggregate
    all_verts = []
    all_faces = []
    vertex_offset = 0
    for center, radius in zip(sphere_xyz_np, sphere_r_np):
        # Create an icosphere mesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        # Move sphere to center
        mesh.apply_translation(center)
        verts = mesh.vertices
        faces = mesh.faces
        all_verts.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if len(all_verts) == 0:
        logger.warning("No spheres to export")
        return

    # Combine all meshes
    verts_combined = np.vstack(all_verts)
    faces_combined = np.vstack(all_faces)

    # Create Trimesh object and export
    tri = trimesh.Trimesh(vertices=verts_combined, faces=faces_combined, process=False)
    tri.export(path)
    logger.info(f"Spheres exported to {path}")


def export_mesh_and_spheres_with_time(mesh: Tuple[torch.Tensor, torch.Tensor], sphere_xyz: torch.Tensor, sphere_r: torch.Tensor):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_path = f"./output/{formatted_time}_mesh.obj"
    spheres_path = f"./output/{formatted_time}_spheres.ply"
    export_mesh(mesh, mesh_path)
    export_spheres(sphere_xyz, sphere_r, spheres_path)



def get_spheres_mesh(sphere_xyz: torch.Tensor, sphere_r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sphere_xyz_np = sphere_xyz.detach().cpu().numpy()
    sphere_r_np = sphere_r.detach().cpu().numpy()

    all_verts = []
    all_faces = []
    vertex_offset = 0
    for center, radius in zip(sphere_xyz_np, sphere_r_np):
        # Create an icosphere mesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        # Move sphere to center
        mesh.apply_translation(center)
        verts = mesh.vertices
        faces = mesh.faces
        all_verts.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if len(all_verts) == 0:
        logger.warning("No spheres to export")
        return  

    # Combine all meshes
    verts_combined = np.vstack(all_verts)
    faces_combined = np.vstack(all_faces)


    return torch.Tensor(verts_combined), torch.Tensor(faces_combined)