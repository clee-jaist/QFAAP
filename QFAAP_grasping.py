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
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

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
    parser.add_argument('--num-iter', type=int, default=1,
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

    # Initialize robot and gripper
    ip = "your robot ip"
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.reset(wait=True)

    arm.set_position(x=200, y=0, z=66.5,
                          roll=-180, pitch=0, yaw=0, speed=150,
                          is_radian=False)

    arm.set_position(x=447.2, y=43, z=780,
                          roll=-180, pitch=0, yaw=0, speed=150,
                          is_radian=False)

    code = arm.set_gripper_enable(True)
    code = arm.set_gripper_mode(0)

    # Initialize depth camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id="your camera ID")
    color_intrinsics = cam.connect()
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
        fig = plt.figure(figsize=(12, 10))
        count = 0

        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']

            # Warm up frame, you can delete it for faster response
            if count < 10:
                cv2.destroyAllWindows()
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)
                    count += 1
                    if count == 1:
                        arm.set_position(x=447.2, y=43, z=780,
                                         roll=-180, pitch=0, yaw=0, speed=150,
                                         is_radian=False)
                        while True:
                            ret, initPose = arm.get_position()
                            # print(initPose)
                            tolerance = 3
                            if abs(initPose[0] - 447.2) < tolerance and \
                                    abs(initPose[1] - 43) < tolerance and \
                                    abs(initPose[2] - 780) < tolerance:
                                break
            else:
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
                original_im = convert_opencv_input(original_im, 'rgb')

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

                    # Original grasp detection, RGB: rgb, Quality: q_img
                    x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                    xc = x.to(device)
                    pred = net.predict(xc)
                    pos = pred['pos']
                    q_img, ang_img, width_img = post_process_output(pos, pred['cos'], pred['sin'], pred['width'])

                    # Original-SZ grasp detection, RGB: rgb (different grasp box), Quality: q_img_1
                    pos_1 = pos.detach().clone()
                    pos_1[masks] = 0
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
                    pos_r1[masks] = 0
                    q_img_r1, ang_img_r1, width_img_r1 = post_process_output(pos_r1, pred_r['cos'],
                                                                             pred_r['sin'], pred_r['width'])

                    #=================================================================================================#

                    # Grasping
                    print("Begin Grasping.....")
                    grasp_point1 = peak_local_max(q_img_r1, min_distance=1, threshold_abs=0.1, num_peaks=1)
                    best_point = [0, 0, 0]
                    top = [0, 0]
                    tlength = 0
                    tangle = 0
                    if len(grasp_point1) > 0:
                        print('grasp points:', len(grasp_point1))
                        for grasp_point in grasp_point1:
                            length = width_img[grasp_point[0], grasp_point[1]]
                            angle = ang_img[grasp_point[0], grasp_point[1]]
                            center = [grasp_point[0], grasp_point[1]]
                            data_array = [center[1] + 208, center[0] + 128]
                            aligned_depth_frame = image_bundle['aligned_depth_frame']
                            dis = aligned_depth_frame.get_distance(data_array[0], data_array[1])
                            x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                      pixel=[data_array[0], data_array[1]],
                                                                      depth=dis)
                            campos = [x, y, z]

                            if 0.1 < z < 0.80:
                                best_point = campos
                                top = grasp_point
                                tlength = length
                                tangle = angle

                        print('best_point:', best_point)
                        print('tlength:', tlength)
                        print('tangle:', -tangle)

                        if best_point == [0, 0, 0]:
                            break

                        else:
                            # Visualize results
                            # plot_results(fig=fig,
                            #              no_grasps = args.n_grasps,
                            #              rgb_img = cam_data.get_rgb(rgb, False),
                            #              grasp_q_img = q_img,
                            #              grasp_q_img_1 = q_img_1,
                            #              grasp_angle_img = ang_img,  # Original and Original-SZ use same angle and width
                            #              grasp_width_img = width_img,
                            #              rgb_img_r = cam_data.get_rgb(rgb_r, False),
                            #              grasp_q_img_r = q_img_r,
                            #              grasp_q_img_r1 = q_img_r1,
                            #              grasp_angle_img_r = ang_img_r,  # QFAAP and Original-NSZ use same angle and width
                            #              grasp_width_img_r = width_img_r)

                            if keyboard.is_pressed('q'):
                                print("break and save")
                                # save_results(
                                #     no_grasps=args.n_grasps,
                                #     rgb_img=cam_data.get_rgb(rgb, False),
                                #     grasp_q_img=q_img,
                                #     grasp_q_img_1=q_img_1,
                                #     grasp_angle_img=ang_img,
                                #     grasp_width_img=width_img,
                                #     rgb_img_r=cam_data.get_rgb(rgb_r, False),
                                #     grasp_q_img_r=q_img_r,
                                #     grasp_q_img_r1=q_img_r1,
                                #     grasp_angle_img_r=ang_img_r,
                                #     grasp_width_img_r=width_img_r)

                                if best_point != None:

                                    tangle = tangle * (180 / np.pi)

                                    # Please use your own calibration results
                                    robot_pos = (best_point[1] + x, y - best_point[0])

                                    arm.set_tool_position(x=robot_pos[0] * 1000, y=robot_pos[1] * 1000,
                                                          z=400,
                                                          roll=0, pitch=0, yaw=-tangle, speed=200,
                                                          is_radian=False)

                                    code = arm.set_gripper_position(tlength * 10, wait=True)

                                    arm.set_tool_position(x=0, y=0,
                                                          z=campos[2] * 1000 - 0.26 * 1000 - 200,
                                                          roll=0, pitch=0, yaw=0, speed=100,
                                                          is_radian=False)
                                    arm.set_tool_position(x=0, y=0,
                                                          z=300,
                                                          roll=0, pitch=0, yaw=0, speed=50,
                                                          is_radian=False)
                                    code = arm.set_gripper_mode(0)
                                    code = arm.set_gripper_position(250, wait=True)
                                    code = arm.set_gripper_position(tlength * 10 + 100, wait=True)
                                    arm.set_tool_position(x=0, y=0,
                                                          z=-100,
                                                          roll=0, pitch=0, yaw=0, speed=50,
                                                          is_radian=False)
                                    arm.set_position(x=423.6, y=-314,
                                                     z=300,
                                                     roll=-180, pitch=0, yaw=0, speed=100,
                                                     is_radian=False)
                                    arm.set_position(x=423.6, y=-314,
                                                          z=167,
                                                          roll=-180, pitch=0, yaw=0, speed=100,
                                                          is_radian=False)
                                    code = arm.set_gripper_position(850, wait=True)
                                    arm.set_position(x=423.6, y=-314,
                                                          z=300,
                                                          roll=-180, pitch=0, yaw=0, speed=100,
                                                          is_radian=False)
                                    arm.set_position(x=447.2, y=43, z=780,
                                                     roll=-180, pitch=0, yaw=0, speed=150,
                                                     is_radian=False)
                                    code = arm.set_gripper_position(0, wait=True)

                                    while True:
                                        ret, initPose = arm.get_position()
                                        tolerance = 3
                                        if abs(initPose[0] - 447.2) < tolerance and \
                                                abs(initPose[1] - 43) < tolerance and \
                                                abs(initPose[2] - 780) < tolerance:
                                            break

                                    count = 0
                                break
                    # =================================================================================================#
    finally:
        print('grasping finished')
        arm.reset(wait=True)