import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
# import spaces         # only for web demo

from pi3.utils.geometry import se3_inverse, homogenize_points, depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from sparsepi3.models.sparsepi3 import SparsePi3
import trimesh
import matplotlib
from scipy.spatial.transform import Rotation
from demo_gradio import run_model, predictions_to_glb



def gradio_demo(
    target_dir,
    result_dir,
    conf_thres=20,
    frame_filter="All",
    show_cam=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    os.makedirs(result_dir, exist_ok=True)
    prediction_save_path = os.path.join(result_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)


    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        result_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing and loading Pi3 model...")

    # model = Pi3.from_pretrained("yyfz233/Pi3")
    # # model = Pi3()
    # # model.load_state_dict(torcdtype = torch.bfloat16h.load('ckpts/pi3.pt', weights_only=False, map_location=device))

    # model.eval()
    
    
    # model_name = 'SparsePi3_fix-after_20th_layers'
    # model_name = 'SparsePi3_fix-swa-4-1'
    # model_name = 'sparsefixpi3'
    model_name = 'pi3'
    # This loads the weights directly into your model instance
    if model_name == 'sparsefixpi3':
        # model = SparsePi3()
        model = SparsePi3()
        ckpt = torch.load('/cis/home/zshao14/Downloads/StreamVGGT/checkpoints/SparsePi3_fix_alpha0.1_lr1e-5_nog/checkpoint-final.pth')
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_name == 'sparsefixswapi3':
        model = SparsePi3()
        ckpt = torch.load('/cis/home/zshao14/Downloads/StreamVGGT/checkpoints/SparsePi3_fixswa_alpha0.1_lr1e-5_nog/checkpoint-final.pth')
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_name == 'sparsepi3':
        model = Pi3()
        ckpt = torch.load('/cis/home/zshao14/Downloads/StreamVGGT/checkpoints/SparsePi3_alpha0.1_lr1e-5_nog/checkpoint-final.pth')
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_name == 'SparsePi3_fix-swa-4-1':
        model = Pi3()
        ckpt = torch.load('/cis/home/zshao14/Downloads/StreamVGGT/checkpoints/SparsePi3_fix-swa-4-1_alpha0.1_lr1e-5_nog/checkpoint-final.pth')
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_name == 'SparsePi3_fix-after_20th_layers':
        model = Pi3()
        ckpt = torch.load('/cis/home/zshao14/Downloads/StreamVGGT/checkpoints/SparsePi3_fix-after_20th_layers_alpha0.1_lr1e-5_nog/checkpoint-final.pth')
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_name == 'pi3':
        model = Pi3.from_pretrained("yyfz233/Pi3")
        # model = Pi3()
        # from safetensors.torch import load_model
        # load_model(model, '/cis/home/zshao14/Downloads/StreamVGGT/ckpt/model.safetensors')
    elif model_name == 'faster_pi3':
        from sparse_vggt.models.pi3 import sparse_model_from_pi3
        model = Pi3.from_pretrained("yyfz233/Pi3")
        model, aux_output_store = sparse_model_from_pi3(model, sparse_ratio=0.5, cdf_threshold=None)
    
    model = model.eval()
    model = model.to(device)
    gradio_demo('/cis/home/zshao14/projects/wriva/other_dashcam/bdd100k_6', f'/cis/home/zshao14/projects/wriva/other_dashcam/bdd100k_6/Pi3')
    gradio_demo('/cis/home/zshao14/projects/wriva/other_dashcam/dcvr2', f'/cis/home/zshao14/projects/wriva/other_dashcam/dcvr2/Pi3')
    gradio_demo('/cis/home/zshao14/projects/wriva/other_dashcam/dcvr3', f'/cis/home/zshao14/projects/wriva/other_dashcam/dcvr3/Pi3')
    gradio_demo('/cis/home/zshao14/projects/wriva/other_dashcam/dcvr9', f'/cis/home/zshao14/projects/wriva/other_dashcam/dcvr9/Pi3')
    # gradio_demo('/cis/home/zshao14/Documents/dpp/mason/day', f'/cis/home/zshao14/Documents/dpp/mason/day/{model_name}')
    # gradio_demo('/cis/home/zshao14/Documents/dpp/mason/day', f'/cis/home/zshao14/Documents/dpp/mason/day/{model_name}')
    # gradio_demo('/cis/home/zshao14/Documents/dpp/heinly2014/ToH_20', f'/cis/home/zshao14/Documents/dpp/heinly2014/ToH_20/{model_name}_test')
    # gradio_demo('/cis/home/zshao14/wriva/mc-sparse-folder/t01_v51_s08_r01_ImageDensity_STR0067/input', f'/cis/home/zshao14/wriva/mc-sparse-folder/t01_v51_s08_r01_ImageDensity_STR0067/{model_name}-80')
    # gradio_demo('/cis/home/zshao14/wriva/mc-sparse-folder/t01_v19_s04_r01_ImageDensity_S07_Cathedral_Indoor/input', f'/cis/home/zshao14/wriva/mc-sparse-folder/t01_v19_s04_r01_ImageDensity_S07_Cathedral_Indoor/{model_name}-80')
    # gradio_demo('/cis/home/zshao14/wriva/mc-sparse-folder/t01_v17_s04_r01_ImageDensity_M09_Water/input', f'/cis/home/zshao14/wriva/mc-sparse-folder/t01_v17_s04_r01_ImageDensity_M09_Water/{model_name}-50')
    # gradio_demo('/cis/home/zshao14/wriva/mc-sparse-folder/t07_v10_s06_r01_EnvironmentalVariation_A03_Season_Remsen_Courtyard/input', f'/cis/home/zshao14/wriva/mc-sparse-folder/t07_v10_s06_r01_EnvironmentalVariation_A03_Season_Remsen_Courtyard/{model_name}-80')
    
    # gradio_demo('/cis/home/zshao14/Documents/dpp/mason/input', f'/cis/home/zshao14/Documents/dpp/mason/input/{model_name}')
    
    # gradio_demo('/cis/home/zshao14/Documents/dpp/yan2017/oats', f'/cis/home/zshao14/Documents/dpp/yan2017/oats/{model_name}_swa5')
    # root_dir = '/cis/home/zshao14/Documents/dpp/heinly2014'
    # for folder in os.listdir(root_dir):
    #     if folder == 'berliner_dom':
    #         continue
    #     result_dir = os.path.join(root_dir, folder, f'{model_name}_dpp')
    #     os.makedirs(result_dir, exist_ok=True)
    #     gradio_demo(os.path.join(root_dir, folder), result_dir)