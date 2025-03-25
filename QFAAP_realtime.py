import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch.nn.functional as F
import numpy as np
import torch.utils.data
import time
import cv2
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from PIL import Image
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
from scipy.ndimage import binary_fill_holes
from hand_seg.image_utils import convert_opencv_input
from hand_seg.model import DeepLabModel
import matplotlib.pyplot as plt
import keyboard

logging.basicConfig(level=logging.INFO)


def parse_args():

    # Grasp Detection
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default ='OCID_GRcovnet_RGB_epoch_40_iou_0.54',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')

    # PQGD
    parser.add_argument('--pur-file', type=str, default='OCID_epoch_41_AQP_0.9226_PQGD_0.9366.PNG',
                        help='original perturbation file')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Maximum perturbation')
    parser.add_argument('--alpha', type=float, default=0.008,
                        help='Step size for perturbation')
    parser.add_argument('--num-iter', type=int, default=5,
                        help='Number of iterations for perturbation')

    # Hand Segmentation
    parser.add_argument('--model-name', type=str, default='model_08_05_21',
                        help='model name.')
    parser.add_argument('--model-dir', type=str, default='Hand_seg_models',
                        help='model path.')

    # Device
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


def quality_pgd(model, inputs, epsilon, alpha, num_iter, cropped_mask, ori_img, device):

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

        loss.backward()

        # Compute perturbation
        adv = perturbed_inputs - alpha * perturbed_inputs.grad.sign()
        perturbation = torch.clamp(adv - ori_img, min=-epsilon, max=epsilon)

        min = ori_img.min()
        max = ori_img.max()
        # Apply perturbation only to the masked region
        cropped_mask = cropped_mask.expand(1, 3, 224, 224)
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

if __name__ == '__main__':

    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=943222070907)
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load grasp detection model
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Load hand segmentation model
    model_path = os.path.join(args.model_dir, args.model_name)
    h_model = DeepLabModel(model_path, 360, 360)

    # Load original patch
    logging.info('Loading patch...')
    perturbation = Image.open(args.pur_file)

    try:
        fig = plt.figure(figsize=(10, 5))

        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']

            # Preprocessing RGB frame
            pil_rgb = Image.fromarray(rgb)
            width, height = pil_rgb.size
            left = (width - 224) / 2
            top = (height - 224) / 2
            right = (width + 224) / 2
            bottom = (height + 224) / 2
            cropped_pic = pil_rgb.crop((left, top, right, bottom))
            original_im = np.array(cropped_pic)
            original_im = cv2.cvtColor(original_im, cv2.COLOR_RGB2BGR)
            original_im = convert_opencv_input(original_im,  'rgb')

            # Hand segmentation
            masks = h_model.run(original_im)
            masks = masks.astype(np.uint8)
            masks = cv2.resize(masks, (224, 224))

            # Extend mask to image size
            masks = masks.astype(np.uint8)
            masks = binary_fill_holes(masks).astype(np.uint8)
            image_size = (480, 640)
            expanded_mask = np.zeros(image_size, dtype=np.uint8)
            center_x = (image_size[0] - masks.shape[0]) // 2
            center_y = (image_size[1] - masks.shape[1]) // 2
            expanded_mask[center_x:center_x + masks.shape[0], center_y:center_y + masks.shape[1]] = masks
            masks2 = expanded_mask

            # Process patch
            new_height, new_width = 480, 640
            t_perturbation = np.ones((480, 640, 3), dtype=np.float32)
            t_perturbation[128:352, 208:432] = perturbation

            # Add the patch to the center area
            rgb_r = rgb.copy()
            rgb_r[masks2 == 1] = t_perturbation[masks2 == 1]

            with torch.no_grad():

                # Mask process
                masks = torch.tensor(masks, dtype=torch.bool)
                masks = masks.unsqueeze(0).unsqueeze(0).expand(1, 3, 224, 224)
                masks = masks[:, :1, :, :]
                kernel_size = 3
                dilated_mask = F.max_pool2d(masks.float(), kernel_size=kernel_size, stride=1, padding=0) > 0
                dilated_mask = F.pad(dilated_mask, (1, 1, 1, 1), value=True)

                # Original grasp detection, RGB: rgb, Quality: q_img
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                xc = x.to(device)
                pred = net.predict(xc)
                pos = pred['pos']
                q_img, ang_img, width_img = post_process_output(pos, pred['cos'], pred['sin'], pred['width'])

                # Original-SZ grasp detection, RGB: rgb (different grasp box), Quality: q_img_1
                pos_1 = pos.detach().clone()
                pos_1[dilated_mask] = 0
                q_img_1, ang_img_1, width_img_1 = post_process_output(pos_1, pred['cos'], pred['sin'], pred['width'])

                # QFAAP-NSZ grasp detection, RGB: rgb_r, Quality: q_img_r
                x_r, depth_img_r, rgb_img_r = cam_data.get_data(rgb=rgb_r, depth=depth)
                xc_r = x_r.to(device)
                ori_img = xc_r.detach().clone()
                xc_r_p = quality_pgd(net, xc_r, args.epsilon, args.alpha, args.num_iter, masks, ori_img, device)
                pred_r = net.predict(xc_r_p)
                pos_r = pred_r['pos']
                q_img_r, ang_img_r, width_img_r = post_process_output(pos_r, pred_r['cos'],
                                                                      pred_r['sin'], pred_r['width'])

                # QFAAP grasp detection, RGB: rgb_r (different grasp box), Quality: q_img_r1
                pos_r1 = pos_r.detach().clone()
                pos_r1[dilated_mask] = 0
                q_img_r1, ang_img_r1, width_img_r1 = post_process_output(pos_r1, pred_r['cos'],
                                                                         pred_r['sin'], pred_r['width'])

                # Visualize results
                plot_results(fig=fig,
                             no_grasps = args.n_grasps,
                             rgb_img = cam_data.get_rgb(rgb, False),
                             grasp_q_img = q_img,
                             grasp_q_img_1 = q_img_1,
                             grasp_angle_img = ang_img,  # Original and Original-SZ use same angle and width
                             grasp_width_img = width_img,
                             rgb_img_r = cam_data.get_rgb(rgb_r, False),
                             grasp_q_img_r = q_img_r,
                             grasp_q_img_r1 = q_img_r1,
                             grasp_angle_img_r = ang_img_r,  # QFAAP and Original-NSZ use same angle and width
                             grasp_width_img_r = width_img_r)

                if keyboard.is_pressed('q'):
                    print("break and save")
                    save_results(
                        no_grasps=args.n_grasps,
                        rgb_img=cam_data.get_rgb(rgb, False),
                        grasp_q_img=q_img,
                        grasp_q_img_1=q_img_1,
                        grasp_angle_img=ang_img,
                        grasp_width_img=width_img,
                        rgb_img_r=cam_data.get_rgb(rgb_r, False),
                        grasp_q_img_r=q_img_r,
                        grasp_q_img_r1=q_img_r1,
                        grasp_angle_img_r=ang_img_r,
                        grasp_width_img_r=width_img_r)
                    break

    finally:
        None