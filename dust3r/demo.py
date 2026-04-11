# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False,
                                 save_glb=True,
                                 save_ply=True,
                                 align_ground_to_oxz_export=True,
                                 tsdf_fusion=False,
                                 view_consistent_merge=False,
                                 vcm_voxel_size=0.005,
                                 wall_slab=False,
                                 slab_scale_duoi=0.02,
                                 slab_scale_tren=0.18,
                                 wall_collapse_source="all",
                                 wall_collapse_strength=1.0,
                                 wall_harmonize_parallel_planes=True,
                                 wall_harmonize_parallel_tol_deg=4.0,
                                 wall_harmonize_offset_tol_ratio=0.004,
                                 remove_ground=False,
                                 ground_y_preference="low",
                                 ground_coplanar_angle_tol_deg=6.0,
                                 ground_coplanar_offset_tol=-1.0,
                                 remove_outlier_cc=False,
                                 statistic_plane=False,
                                 sp_normal_variance_threshold_deg=60.0,
                                 sp_coplanarity_deg=75.0,
                                 sp_outlier_ratio=0.75,
                                 sp_min_num_points=100,
                                 sp_normal_alignment_threshold=0.92,
                                 sp_robust_sigma_k=2.5,
                                 sp_flatten_distance_threshold=-1.0,
                                 precomputed_points=None,
                                 precomputed_colors=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    slab_cut_lines = None
    facade_pts = None
    facade_col = None

    # full pointcloud (also used for PLY export)
    if precomputed_points is not None:
        pts = np.asarray(precomputed_points)
        if precomputed_colors is not None:
            col = np.asarray(precomputed_colors)
        else:
            col = np.zeros((len(pts), 3), dtype=np.float32)
    else:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])

    if as_pointcloud and view_consistent_merge:
        from dust3r.utils.custom import view_consistent_merge as _view_consistent_merge

        vcm_res = _view_consistent_merge(
            pts.reshape(-1, 3),
            col.reshape(-1, 3),
            enabled=True,
            voxel_size=float(vcm_voxel_size),
        )
        pts = vcm_res.points
        col = vcm_res.colors if vcm_res.colors is not None else col

    if as_pointcloud and (not remove_ground) and align_ground_to_oxz_export:
        from dust3r.utils.custom import align_pointcloud_to_ground_oxz as _align_pointcloud_to_ground_oxz

        align_res = _align_pointcloud_to_ground_oxz(
            pts.reshape(-1, 3),
            col.reshape(-1, 3),
            enabled=True,
            strict=False,
        )
        pts = align_res.points
        col = align_res.colors if align_res.colors is not None else col

    if remove_ground:
        # Optional dependency: only used when enabled.
        from dust3r.utils.custom import remove_ground as _remove_ground

        coplanar_offset_tol = None if float(ground_coplanar_offset_tol) < 0 else float(ground_coplanar_offset_tol)
        res = _remove_ground(
            pts.reshape(-1, 3),
            col.reshape(-1, 3),
            enabled=True,
            regenerate_ground=False,
            align_ground_to_oxz=bool(align_ground_to_oxz_export),
            y_preference=str(ground_y_preference),
            coplanar_angle_tol_deg=float(ground_coplanar_angle_tol_deg),
            coplanar_offset_tol=coplanar_offset_tol,
            strict=False,
        )
        pts = res.points
        col = res.colors

    if remove_outlier_cc:
        # CloudCompare-like connected components filtering (keep largest cluster).
        from dust3r.utils.custom import remove_outlier_cc as _remove_outlier_cc

        res = _remove_outlier_cc(
            pts.reshape(-1, 3),
            col.reshape(-1, 3),
            enabled=True,
            octree_level=8,
            min_points_per_component=20,
            keep_largest_only=True,
        )
        pts = res.points
        col = res.colors if res.colors is not None else col

    if as_pointcloud and wall_slab:
        from dust3r.utils.custom import wall_slab_grid_from_points as _wall_slab_grid_from_points

        slab_res = _wall_slab_grid_from_points(
            pts.reshape(-1, 3),
            col.reshape(-1, 3),
            align_ground=False,
            scale_duoi=float(slab_scale_duoi),
            scale_tren=float(slab_scale_tren),
            collapse_layers=True,
            collapse_source=str(wall_collapse_source),
            collapse_strength=float(wall_collapse_strength),
            harmonize_parallel_planes=bool(wall_harmonize_parallel_planes),
            harmonize_parallel_tol_deg=float(wall_harmonize_parallel_tol_deg),
            harmonize_offset_tol_ratio=float(wall_harmonize_offset_tol_ratio),
            strict=False,
        )
        # Keep cloud unchanged; wall slab is used to compute/visualize cut range only.
        slab_cut_lines = slab_res.cut_lines
        if len(slab_res.facade_collapsed_points) > 0:
            facade_pts = slab_res.facade_collapsed_points
            if slab_res.colors is not None:
                if slab_res.source_mask.shape[0] == slab_res.colors.shape[0]:
                    src_cols = slab_res.colors[slab_res.source_mask]
                else:
                    src_cols = slab_res.colors
                facade_col = src_cols

    # statistic_plane is temporarily hidden/disabled.
    # To re-enable, uncomment the block below and restore UI controls/events.
    # if statistic_plane:
    #     # Optional dependency: only used when enabled.
    #     from dust3r.utils.custom import statistic_plane as _statistic_plane
    #
    #     flatten_thr = None if float(sp_flatten_distance_threshold) <= 0 else float(sp_flatten_distance_threshold)
    #     res = _statistic_plane(
    #         pts.reshape(-1, 3),
    #         col.reshape(-1, 3),
    #         enabled=True,
    #         normal_variance_threshold_deg=float(sp_normal_variance_threshold_deg),
    #         coplanarity_deg=float(sp_coplanarity_deg),
    #         outlier_ratio=float(sp_outlier_ratio),
    #         min_num_points=int(sp_min_num_points),
    #         normal_alignment_threshold=float(sp_normal_alignment_threshold),
    #         robust_sigma_k=float(sp_robust_sigma_k),
    #         flatten_distance_threshold=flatten_thr,
    #         strict=False,
    #     )
    #     pts = res.points
    #     col = res.colors if res.colors is not None else col

    if as_pointcloud:
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
        if slab_cut_lines is not None and len(slab_cut_lines) > 0:
            scene.add_geometry(trimesh.load_path(slab_cut_lines))
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    preview_scene = scene.copy()
    # Keep pointcloud preview in the exact export frame so visual slab ranges
    # match saved GLB/PLY coordinates.
    if not as_pointcloud:
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        preview_scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # Always keep a GLB preview for the 3D widget.
    preview_glb = os.path.join(outdir, 'scene.glb')
    facade_preview_glb = os.path.join(outdir, 'scene_facade.glb') if (as_pointcloud and wall_slab and facade_pts is not None and len(facade_pts) > 0) else None
    dl_glb = os.path.join(outdir, 'scene_download.glb') if bool(save_glb) else None
    dl_ply = os.path.join(outdir, 'scene_download.ply') if bool(save_ply) else None
    if not silent:
        print('(exporting 3D preview to', preview_glb, ')')
    preview_scene.export(file_obj=preview_glb)

    if facade_preview_glb is not None:
        facade_scene = trimesh.Scene()
        fc = facade_col if facade_col is not None else np.zeros((len(facade_pts), 3), dtype=np.float32)
        facade_scene.add_geometry(trimesh.PointCloud(facade_pts.reshape(-1, 3), colors=fc.reshape(-1, 3)))
        if slab_cut_lines is not None and len(slab_cut_lines) > 0:
            facade_scene.add_geometry(trimesh.load_path(slab_cut_lines))
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(facade_scene, pose_c2w, camera_edge_color,
                          None if transparent_cams else imgs[i], focals[i],
                          imsize=imgs[i].shape[1::-1], screen_width=cam_size)
        facade_scene.export(file_obj=facade_preview_glb)

    if dl_glb is not None:
        # Download GLB keeps original aligned coordinates (no viewer-only transform).
        scene.export(file_obj=dl_glb)
    if dl_ply is not None:
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        pct.export(file_obj=dl_ply)

    return preview_glb, facade_preview_glb, dl_glb, dl_ply


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05,
                            save_glb=True,
                            save_ply=True,
                            align_ground_to_oxz_export=True,
                            tsdf_fusion=False,
                            view_consistent_merge=False,
                            vcm_voxel_size=0.005,
                            wall_slab=False,
                            slab_scale_duoi=0.02,
                            slab_scale_tren=0.18,
                            wall_collapse_source="all",
                            wall_collapse_strength=1.0,
                            wall_harmonize_parallel_planes=True,
                            wall_harmonize_parallel_tol_deg=4.0,
                            wall_harmonize_offset_tol_ratio=0.004,
                            remove_ground=False,
                            ground_y_preference="low",
                            ground_coplanar_angle_tol_deg=6.0,
                            ground_coplanar_offset_tol=-1.0,
                            remove_outlier_cc=False,
                            statistic_plane=False,
                            sp_normal_variance_threshold_deg=60.0,
                            sp_coplanarity_deg=75.0,
                            sp_outlier_ratio=0.75,
                            sp_min_num_points=100,
                            sp_normal_alignment_threshold=0.92,
                            sp_robust_sigma_k=2.5,
                            sp_flatten_distance_threshold=-1.0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None, None, None, None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    depths = to_numpy(scene.get_depthmaps())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    pre_pts = None
    pre_col = None
    if as_pointcloud and tsdf_fusion:
        from dust3r.utils.custom import tsdf_fuse_views as _tsdf_fuse_views

        tsdf_res = _tsdf_fuse_views(
            depths,
            cams2world,
            focals,
            masks=msk,
            images=rgbimg,
            enabled=True,
            strict=False,
        )
        if len(tsdf_res.points) > 0:
            pre_pts = tsdf_res.points
            pre_col = tsdf_res.colors

    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent,
                                        save_glb=save_glb,
                                        save_ply=save_ply,
                                        align_ground_to_oxz_export=align_ground_to_oxz_export,
                                        tsdf_fusion=tsdf_fusion,
                                        view_consistent_merge=view_consistent_merge,
                                        vcm_voxel_size=vcm_voxel_size,
                                        wall_slab=wall_slab,
                                        slab_scale_duoi=slab_scale_duoi,
                                        slab_scale_tren=slab_scale_tren,
                                        wall_collapse_source=wall_collapse_source,
                                        wall_collapse_strength=wall_collapse_strength,
                                        wall_harmonize_parallel_planes=wall_harmonize_parallel_planes,
                                        wall_harmonize_parallel_tol_deg=wall_harmonize_parallel_tol_deg,
                                        wall_harmonize_offset_tol_ratio=wall_harmonize_offset_tol_ratio,
                                        remove_ground=remove_ground,
                                        ground_y_preference=ground_y_preference,
                                        ground_coplanar_angle_tol_deg=ground_coplanar_angle_tol_deg,
                                        ground_coplanar_offset_tol=ground_coplanar_offset_tol,
                                        remove_outlier_cc=remove_outlier_cc,
                                        statistic_plane=statistic_plane,
                                        sp_normal_variance_threshold_deg=sp_normal_variance_threshold_deg,
                                        sp_coplanarity_deg=sp_coplanarity_deg,
                                        sp_outlier_ratio=sp_outlier_ratio,
                                        sp_min_num_points=sp_min_num_points,
                                        sp_normal_alignment_threshold=sp_normal_alignment_threshold,
                                        sp_robust_sigma_k=sp_robust_sigma_k,
                                        sp_flatten_distance_threshold=sp_flatten_distance_threshold,
                                        precomputed_points=pre_pts,
                                        precomputed_colors=pre_col)


def get_3D_model_and_downloads(*args, **kwargs):
    preview_glb, facade_preview_glb, dl_glb, dl_ply = get_3D_model_from_scene(*args, **kwargs)
    return preview_glb, (facade_preview_glb or preview_glb), dl_glb, dl_ply


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, save_glb, save_ply,
                            align_ground_to_oxz_export,
                            tsdf_fusion,
                            view_consistent_merge,
                            vcm_voxel_size,
                            wall_slab,
                            slab_scale_duoi,
                            slab_scale_tren,
                            wall_collapse_source,
                            wall_collapse_strength,
                            wall_harmonize_parallel_planes,
                            wall_harmonize_parallel_tol_deg,
                            wall_harmonize_offset_tol_ratio,
                            remove_ground,
                            ground_y_preference,
                            ground_coplanar_angle_tol_deg,
                            ground_coplanar_offset_tol,
                            remove_outlier_cc,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    try:
        square_ok = model.square_ok
    except Exception as e:
        square_ok = False
    imgs = load_images(filelist, size=image_size, verbose=not silent, patch_size=model.patch_size, square_ok=square_ok)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile, facade_outfile, glbfile, plyfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                        clean_depth, transparent_cams, cam_size,
                                                        save_glb=save_glb,
                                                        save_ply=save_ply,
                                                        align_ground_to_oxz_export=align_ground_to_oxz_export,
                                                        tsdf_fusion=tsdf_fusion,
                                                        view_consistent_merge=view_consistent_merge,
                                                        vcm_voxel_size=vcm_voxel_size,
                                                        wall_slab=wall_slab,
                                                        slab_scale_duoi=slab_scale_duoi,
                                                        slab_scale_tren=slab_scale_tren,
                                                        wall_collapse_source=wall_collapse_source,
                                                        wall_collapse_strength=wall_collapse_strength,
                                                        wall_harmonize_parallel_planes=wall_harmonize_parallel_planes,
                                                        wall_harmonize_parallel_tol_deg=wall_harmonize_parallel_tol_deg,
                                                        wall_harmonize_offset_tol_ratio=wall_harmonize_offset_tol_ratio,
                                                        remove_ground=remove_ground,
                                                        ground_y_preference=ground_y_preference,
                                                        ground_coplanar_angle_tol_deg=ground_coplanar_angle_tol_deg,
                                                        ground_coplanar_offset_tol=ground_coplanar_offset_tol,
                                                        remove_outlier_cc=remove_outlier_cc)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, facade_outfile, imgs, glbfile, plyfile


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_and_downloads, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=150, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                   ("swin: sliding window", "swin"),
                                                   ("oneref: match one image with all", "oneref")],
                                                  value='swin', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                        minimum=1, maximum=1, step=1, visible=True)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=13.0, minimum=1.0, maximum=20, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.001, minimum=0.001, maximum=0.1, step=0.001)
            with gradio.Row():
                save_glb = gradio.Checkbox(value=False, label="Enable GLB download")
                save_ply = gradio.Checkbox(value=False, label="Enable PLY download")
                align_ground_to_oxz_export = gradio.Checkbox(value=True, label="Align export to ground Oxz")
                tsdf_fusion = gradio.Checkbox(value=False, label="TSDF fusion (view-based)")
                view_consistent_merge = gradio.Checkbox(value=False, label="View-consistent merge (1 layer)")
            with gradio.Row():
                vcm_voxel_size = gradio.Number(label="vcm_voxel_size", value=0.005, precision=6)
            with gradio.Row():
                wall_slab = gradio.Checkbox(value=False, label="Wall slab (low wall band)")
                slab_scale_duoi = gradio.Slider(
                    label="slab_scale_duoi", value=0.02, minimum=0.00, maximum=1.00, step=0.01
                )
                slab_scale_tren = gradio.Slider(
                    label="slab_scale_tren", value=0.18, minimum=0.00, maximum=1.00, step=0.01
                )
            with gradio.Row():
                wall_collapse_source = gradio.Dropdown(
                    ["all", "source"],
                    value="all",
                    label="wall_collapse_source",
                    info="all: giữ mật độ cao (toàn cloud), source: chỉ vùng slab/source",
                )
                wall_collapse_strength = gradio.Slider(
                    label="wall_collapse_strength", value=1.0, minimum=0.0, maximum=1.0, step=0.05
                )
            with gradio.Row():
                wall_harmonize_parallel_planes = gradio.Checkbox(
                    value=True,
                    label="wall_harmonize_parallel_planes",
                )
                wall_harmonize_parallel_tol_deg = gradio.Slider(
                    label="wall_harmonize_parallel_tol_deg", value=4.0, minimum=1.0, maximum=12.0, step=0.5
                )
                wall_harmonize_offset_tol_ratio = gradio.Slider(
                    label="wall_harmonize_offset_tol_ratio", value=0.004, minimum=0.001, maximum=0.02, step=0.001
                )
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                remove_ground = gradio.Checkbox(value=False, label="Remove ground (pointcloud)")
                remove_outlier_cc = gradio.Checkbox(value=False, label="Remove outlier CC (keep largest)")
            with gradio.Row():
                ground_y_preference = gradio.Dropdown(
                    ["low", "high"],
                    value="low",
                    label="ground_y_preference",
                    info="low: ưu tiên mặt phẳng thấp theo trục Y, high: ưu tiên mặt phẳng cao",
                )
                ground_coplanar_angle_tol_deg = gradio.Slider(
                    label="ground_coplanar_angle_tol_deg", value=3.0, minimum=1.0, maximum=20.0, step=0.5
                )
                ground_coplanar_offset_tol = gradio.Number(
                    label="ground_coplanar_offset_tol (-1 = auto)", value=-1.0, precision=6
                )
            with gradio.Row():
                # statistic_plane controls are temporarily hidden.
                # Uncomment the blocks below to re-enable.
                # statistic_plane = gradio.Checkbox(value=False, label="Statistic plane (flatten surfaces)")
                pass
            # with gradio.Row():
            #     sp_normal_variance_threshold_deg = gradio.Slider(
            #         label="sp_normal_variance_threshold_deg", value=60.0, minimum=5.0, maximum=90.0, step=1.0
            #     )
            #     sp_coplanarity_deg = gradio.Slider(
            #         label="sp_coplanarity_deg", value=75.0, minimum=5.0, maximum=90.0, step=1.0
            #     )
            #     sp_outlier_ratio = gradio.Slider(
            #         label="sp_outlier_ratio", value=0.75, minimum=0.0, maximum=1.0, step=0.01
            #     )
            # with gradio.Row():
            #     sp_min_num_points = gradio.Number(
            #         label="sp_min_num_points", value=100, precision=0, minimum=10, maximum=100000
            #     )
            #     sp_normal_alignment_threshold = gradio.Slider(
            #         label="sp_normal_alignment_threshold", value=0.92, minimum=0.5, maximum=0.999, step=0.001
            #     )
            #     sp_robust_sigma_k = gradio.Slider(
            #         label="sp_robust_sigma_k", value=2.5, minimum=0.5, maximum=6.0, step=0.1
            #     )
            #     sp_flatten_distance_threshold = gradio.Number(
            #         label="sp_flatten_distance_threshold (-1 = auto)", value=-1.0, precision=6
            #     )

            export_inputs = [
                scene, min_conf_thr, as_pointcloud, mask_sky,
                clean_depth, transparent_cams, cam_size, save_glb, save_ply, align_ground_to_oxz_export, tsdf_fusion,
                view_consistent_merge, vcm_voxel_size,
                wall_slab, slab_scale_duoi, slab_scale_tren,
                wall_collapse_source, wall_collapse_strength,
                wall_harmonize_parallel_planes, wall_harmonize_parallel_tol_deg, wall_harmonize_offset_tol_ratio,
                remove_ground,
                ground_y_preference,
                ground_coplanar_angle_tol_deg,
                ground_coplanar_offset_tol,
                remove_outlier_cc,
            ]

            outmodel = gradio.Model3D()
            outmodel_facade = gradio.Model3D(label="Facade Band Preview (Multi-layer Reduced)")
            outgallery = gradio.Gallery(label='rgb,depth,confidence', columns=3, height="100%")
            out_glb_file = gradio.File(label="Download scene.glb")
            out_ply_file = gradio.File(label="Download scene.ply")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                      inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                          mask_sky, clean_depth, transparent_cams, cam_size, save_glb, save_ply,
                          align_ground_to_oxz_export, tsdf_fusion,
                          view_consistent_merge, vcm_voxel_size,
                          wall_slab, slab_scale_duoi, slab_scale_tren,
                          wall_collapse_source, wall_collapse_strength,
                          wall_harmonize_parallel_planes, wall_harmonize_parallel_tol_deg, wall_harmonize_offset_tol_ratio,
                          remove_ground,
                          ground_y_preference,
                          ground_coplanar_angle_tol_deg,
                          ground_coplanar_offset_tol,
                          remove_outlier_cc,
                                  scenegraph_type, winsize, refid],
                          outputs=[scene, outmodel, outmodel_facade, outgallery, out_glb_file, out_ply_file])
            min_conf_thr.release(fn=model_from_scene_fun,
                         inputs=export_inputs,
                                 outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            cam_size.change(fn=model_from_scene_fun,
                        inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            save_glb.change(fn=model_from_scene_fun,
                            inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            save_ply.change(fn=model_from_scene_fun,
                            inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            align_ground_to_oxz_export.change(fn=model_from_scene_fun,
                                              inputs=export_inputs,
                                              outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            tsdf_fusion.change(fn=model_from_scene_fun,
                               inputs=export_inputs,
                               outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            view_consistent_merge.change(fn=model_from_scene_fun,
                                         inputs=export_inputs,
                                         outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            vcm_voxel_size.change(fn=model_from_scene_fun,
                                  inputs=export_inputs,
                                  outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_slab.change(fn=model_from_scene_fun,
                         inputs=export_inputs,
                         outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            slab_scale_duoi.release(fn=model_from_scene_fun,
                            inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            slab_scale_tren.release(fn=model_from_scene_fun,
                            inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_collapse_source.change(fn=model_from_scene_fun,
                        inputs=export_inputs,
                        outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_collapse_strength.release(fn=model_from_scene_fun,
                           inputs=export_inputs,
                           outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_harmonize_parallel_planes.change(fn=model_from_scene_fun,
                                  inputs=export_inputs,
                                  outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_harmonize_parallel_tol_deg.release(fn=model_from_scene_fun,
                                    inputs=export_inputs,
                                    outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            wall_harmonize_offset_tol_ratio.release(fn=model_from_scene_fun,
                                    inputs=export_inputs,
                                    outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            as_pointcloud.change(fn=model_from_scene_fun,
                         inputs=export_inputs,
                                 outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            mask_sky.change(fn=model_from_scene_fun,
                        inputs=export_inputs,
                            outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            clean_depth.change(fn=model_from_scene_fun,
                           inputs=export_inputs,
                               outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            transparent_cams.change(model_from_scene_fun,
                            inputs=export_inputs,
                                    outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            remove_ground.change(fn=model_from_scene_fun,
                                inputs=export_inputs,
                                outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            ground_y_preference.change(fn=model_from_scene_fun,
                                       inputs=export_inputs,
                                       outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            ground_coplanar_angle_tol_deg.release(fn=model_from_scene_fun,
                                                  inputs=export_inputs,
                                                  outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            ground_coplanar_offset_tol.change(fn=model_from_scene_fun,
                                              inputs=export_inputs,
                                              outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            remove_outlier_cc.change(fn=model_from_scene_fun,
                                     inputs=export_inputs,
                                     outputs=[outmodel, outmodel_facade, out_glb_file, out_ply_file])
            # statistic_plane events are temporarily hidden.
            # statistic_plane.change(fn=model_from_scene_fun,
            #                     inputs=export_inputs,
            #                     outputs=outmodel)
            # sp_normal_variance_threshold_deg.release(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_coplanarity_deg.release(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_outlier_ratio.release(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_min_num_points.change(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_normal_alignment_threshold.release(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_robust_sigma_k.release(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
            # sp_flatten_distance_threshold.change(fn=model_from_scene_fun, inputs=export_inputs, outputs=outmodel)
    demo.launch(share=False, server_name=server_name, server_port=server_port)



