import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch.utils.data
from PIL import Image
import random
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData2
from utils.visualisation.plot import save_results2
logging.basicConfig(level=logging.INFO)
import time
import torchvision.transforms.functional as TF

def parse_args():

    # Grasp detection and AQP parameters
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default ='QFAAP_models/Cornell_GGCNN_RGB_epoch_47_iou_0.88',
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='E:/cornell_dataset/03/pcd0317r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='E:/cornell_dataset/03/pcd0317d.tiff',
                        help='Depth Image path')
    parser.add_argument('--patch_path', type=str, default='epoch_50_AQP_0.9385_PQGD_0.9590.png',
                        help='AQP Patch path')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=1,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    # PQGD parameters
    parser.add_argument('--epsilon', type=float, default=8/255, help='Maximum perturbation')
    parser.add_argument('--alpha', type=float, default=0.1, help='Step size for perturbation')
    parser.add_argument('--num-iter', type=int, default=5, help='Number of iterations for perturbation')

    args = parser.parse_args()
    return args


def quality_pgd(model, inputs, epsilon, alpha, num_iter, cropped_mask, ori_img, device):
    """
    Perform PGD attack on the given inputs.

    Args:
        model: The target model.
        inputs: The input tensor.
        epsilon: Maximum perturbation.
        alpha: Step size.
        num_iter: Number of iterations.
        cropped_mask: Binary mask indicating the region to perturb.
        ori_img: Original input image.
        device: Compute device.

    Returns:
        Perturbation applied to inputs.
    """
    perturbed_inputs = inputs.clone().detach().to(device)
    cropped_mask = cropped_mask.to(device)

    perturbed_inputs.requires_grad_(True)

    start_time = time.time()

    for _ in range(num_iter):

        with torch.enable_grad():
            model.eval()
            model.zero_grad()

            # Loss
            outputs = model.predict(perturbed_inputs)
            pos_min = torch.min(outputs['pos'])
            pos_max = torch.max(outputs['pos'])
            normalized_pos = (outputs['pos'] - pos_min) / (pos_max - pos_min)
            pos = torch.clamp(normalized_pos, 0, 1)

            pos_patch_values = pos * cropped_mask
            pos_mean = torch.mean(pos_patch_values)
            loss = -pos_mean
            print(loss)
        loss.backward()

        # Compute perturbation
        adv = perturbed_inputs - alpha * perturbed_inputs.grad.sign()
        perturbation = torch.clamp(adv - ori_img, min=-epsilon, max=epsilon)

        # Apply perturbation only to the masked region
        min = ori_img.min()  # Note: the ori_img value range here is not 0-1
        max = ori_img.max()
        cropped_mask = cropped_mask.expand(1, 3, 300, 300)
        perturbed_inputs = ori_img.clone()
        perturbed_inputs[cropped_mask == 1] = torch.clamp(
            ori_img + perturbation, min=min, max=max
        )[cropped_mask == 1]

        # Detach and re-enable gradient tracking
        perturbed_inputs = perturbed_inputs.detach_()
        perturbed_inputs.requires_grad_(True)


    end_time = time.time()

    total_time = end_time - start_time
    print(f"PGD iteration time: {total_time:.4f} seconds")

    return perturbed_inputs


def rescale_patch(patch, min_scale=0.5, max_scale=3, scale=None):
    if not isinstance(patch, Image.Image):
        raise TypeError("The patch must be a PIL image.")

    if scale is not None:
        scale_factor = scale
    else:
        scale_factor = random.uniform(min_scale, max_scale)

    new_size = (int(patch.size[0] * scale_factor), int(patch.size[1] * scale_factor))
    rescaled_patch = patch.resize(new_size, Image.BILINEAR)

    return rescaled_patch, scale_factor


def apply_patch(image, patch, input_size, min_scale=0.5, max_scale=1.5, scale=None):

    patch, scale_factor = rescale_patch(patch, min_scale, max_scale, scale=scale)
    rescaled_patch_size = patch.size[1]

    if input_size <= rescaled_patch_size:
        x_offset = np.zeros(1, dtype=int)
        y_offset = np.zeros(1, dtype=int)
    else:
        x_offset = np.random.randint(0, input_size - rescaled_patch_size, size=1)
        y_offset = np.random.randint(0, input_size - rescaled_patch_size, size=1)

    patch_mask = Image.new("L", image.size, 0)
    patch_mask.paste(1, (x_offset[0], y_offset[0], x_offset[0] + rescaled_patch_size, y_offset[0] + rescaled_patch_size))

    patched_image = image.copy()
    patched_image.paste(patch, (x_offset[0], y_offset[0]))

    return x_offset, y_offset, patched_image, patch_mask

if __name__ == '__main__':
    args = parse_args()

    # Load image
    logging.info('Loading image...')
    rgb = Image.open(args.rgb_path, 'r')
    width, height = rgb.size
    left = (width - 300) / 2
    top = (height - 300) / 2
    right = (width + 300) / 2
    bottom = (height + 300) / 2
    rgb = rgb.crop((left, top, right, bottom))

    depth = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(depth), axis=2)

    # Load patch
    logging.info('Loading patch...')
    patch = Image.open(args.patch_path)

    # Apply patch to rgb image
    x_offset, y_offset, patched_images, patch_mask = apply_patch(rgb, patch, 300,
                                                                 min_scale=0.1, max_scale=1, scale=0.3)

    # Postprocessing image
    rgb = patched_images
    img_data = CameraData2(include_depth=args.use_depth, include_rgb=args.use_rgb)
    rgb = np.array(rgb)
    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    with torch.no_grad():

        # AQP grasp detection, RGB: rgb, Quality: q_img
        xc = x.to(device)
        pred = net.predict(xc)
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        # (AQP+PQGD) grasp detection, RGB: rgb_r, Quality: q_img_r
        ori_img = xc.data
        patch_mask = np.array(patch_mask)
        patch_mask = torch.tensor(patch_mask, dtype=torch.bool)
        patch_mask = patch_mask.unsqueeze(0)

        xc_p = quality_pgd(net, xc, args.epsilon, args.alpha, args.num_iter, patch_mask, ori_img, device)

        # Reverse normalization
        rgb_img_r = xc_p.detach().cpu().squeeze(0)
        rgb_img_r = rgb_img_r * 255.0 + rgb.mean()
        rgb_img_r = rgb_img_r.clamp(0, 255).byte()
        rgb_img_r = TF.to_pil_image(rgb_img_r)

        pred_r = net.predict(xc_p)
        q_img_r, ang_img_r, width_img_r = post_process_output(pred_r['pos'], pred_r['cos'], pred_r['sin'], pred_r['width'])

        if args.save:
            save_results2(
                no_grasps=args.n_grasps,
                rgb_img=img_data.get_rgb(rgb, False),
                rgb_img_r=rgb_img_r,
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                grasp_width_img=width_img,
                grasp_q_img_r=q_img_r,
                grasp_angle_img_r=ang_img_r,
                grasp_width_img_r=width_img_r,
                point=None)
        else:
            None

