import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch.nn.functional as F
import pyrealsense2 as rs
import torch.utils.data
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
import keyboard
import time
from skimage.feature import peak_local_max
from PIL import Image
import cv2
from scipy.ndimage import binary_fill_holes
from hand_seg.image_utils import convert_opencv_input
from hand_seg.model import DeepLabModel
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
logging.basicConfig(level=logging.INFO)
import sys

sys.path.append("..")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from xarm.wrapper import XArmAPI

# Global parameters
MAX_VELO_X = 5  # mm/s 5
MAX_VELO_Y = 20  # mm/s 20
MAX_VELO_Z = 50  # mm/s 30
MAX_ANGULAR_VELO = 0  # degree/s 3
POSITION_TOLERANCE = 300
ORIENTATION_TOLERANCE = 1.0
MIN_DEPTH = 0.1
MAX_DEPTH = 0.820
SERVO_ENABLED = False
CURRENT_POSITION = [355.4, -8.3, 569.9]
CURRENT_ORIENTATION = [0, 0, 0]  # roll, pitch, yaw
TARGET_POSITION = [0, 0, 0]
TARGET_ORIENTATION = [0, 0, 0]
GRIP_WIDTH = 0
CURRENT_DEPTH = 0
CURRENT_VELOCITY = [0, 0, 0, 0, 0, 0]  # vx, vy, vz, vroll, vpitch, vyaw


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


def pqgd(model, inputs, epsilon, alpha, num_iter, cropped_mask, ori_img, device):

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


def initialize_robot(ip='your robot IP'):

    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.reset(wait=True)

    arm.set_position(x=200, y=0, z=66.5,
                     roll=-180, pitch=0, yaw=0, speed=100,
                     is_radian=False)
    arm.set_position(x=447.2, y=43, z=750,
                     roll=-180, pitch=0, yaw=0, speed=100,
                     is_radian=False)

    while True:
        ret, current_pose = arm.get_position()
        if ret == 0:
            tolerance = 3
            if (abs(current_pose[0] - 447.2) < tolerance and
                    abs(current_pose[1] - 43) < tolerance and
                    abs(current_pose[2] - 750) < tolerance):
                break
        time.sleep(0.1)

    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_position(850, wait=True)

    return arm

def velocity_control_loop(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                          h_model, epsilon, alpha, num_iter, perturbation, n_grasps, fig, axes):

    global SERVO_ENABLED, CURRENT_VELOCITY

    # Control frequency
    control_rate = 100  # Hz
    dt = 1.0 / control_rate

    # frame_count = 0

    while True:
        start_time = time.time()

        # if keyboard.is_pressed('q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit...")
            break

        # Get image
        image_bundle = cam.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']

        # # Warm-up frame
        # if frame_count < 3:
        #     frame_count += 1
        #     time.sleep(dt)
        #     continue

        if SERVO_ENABLED:
            # Grasp detection
            best_grasp = process_grasp_detection(cam_data, net, device, rgb, depth, image_bundle,
                                                 color_intrinsics, pose_averager, h_model, epsilon,
                                                 alpha, num_iter, perturbation, n_grasps, fig, axes)
            if best_grasp:
                # Update pose
                update_target_pose(best_grasp)

                #  Calculate and send velocities
                calculate_velocities(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                                     h_model, epsilon, alpha, num_iter, perturbation, n_grasps, fig, axes)
                send_velocity_commands(arm)

            else:
                # No object and stop move
                CURRENT_VELOCITY = [0, 0, 0, 0, 0, 0]
                send_velocity_commands(arm)

        # Keep frequency
        elapsed = time.time() - start_time
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)


def process_grasp_detection(cam_data, net, device, rgb, depth, image_bundle,
                                                 color_intrinsics, pose_averager, h_model, epsilon,
                                                 alpha, num_iter, perturbation, n_grasps, fig, axes):

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
        xc_r_p = pqgd(net, xc_r, epsilon, alpha, num_iter, masks, ori_img, device)
        pred_r = net.predict(xc_r_p)
        pos_r = pred_r['pos']
        q_img_r, ang_img_r, width_img_r = post_process_output(pos_r, pred_r['cos'],
                                                              pred_r['sin'], pred_r['width'])

        # QFAAP grasp detection, RGB: rgb_r (different grasp box), Quality: q_img_r1
        pos_r1 = pos_r.detach().clone()
        pos_r1[masks] = 0
        q_img_r1, ang_img_r1, width_img_r1 = post_process_output(pos_r1, pred_r['cos'],
                                                                 pred_r['sin'], pred_r['width'])

        grasp_points = peak_local_max(q_img_r1, min_distance=0, threshold_abs=0.1, num_peaks=1)

        best_grasp = None
        if len(grasp_points) > 0:
            for grasp_point in grasp_points:
                length = width_img[grasp_point[0], grasp_point[1]]
                angle = ang_img[grasp_point[0], grasp_point[1]]
                center = [grasp_point[0], grasp_point[1]]
                data_array = [center[1] + 208, center[0] + 128]

                aligned_depth_frame = image_bundle['aligned_depth_frame']
                dis = aligned_depth_frame.get_distance(data_array[0], data_array[1])

                if dis > 0:
                    x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                              pixel=[data_array[0], data_array[1]],
                                                              depth=dis)

                    if MIN_DEPTH < z < MAX_DEPTH:  # Safe depth range

                        # Visualization
                        plot_results(fig=fig,
                                     axes=axes,
                                     no_grasps=n_grasps,
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

                        # Filtering
                        smoothed_pos = pose_averager.update(np.array([x, y, z, angle]))

                        # Transform to end-effector coordinate
                        best_grasp = {
                            'position': [smoothed_pos[0]+ your_calib_x, your_calib_y - smoothed_pos[0],
                                         smoothed_pos[2] - your_calib_z],
                            'angle': smoothed_pos[3] * (180 / np.pi),
                            'width': length,
                            'depth': z
                        }
                        break

        print(best_grasp)

        return best_grasp


def update_target_pose(best_grasp):

    global TARGET_POSITION, TARGET_ORIENTATION

    TARGET_POSITION = [
        best_grasp['position'][0] * 1000,  # Transform into mm
        best_grasp['position'][1] * 1000,
        best_grasp['position'][2] * 1000
    ]
    TARGET_ORIENTATION = [0, 0, best_grasp['angle']]


def calculate_velocities(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                         h_model, epsilon, alpha, num_iter, perturbation, n_grasps, fig, axes):

    global CURRENT_VELOCITY, TARGET_POSITION, TARGET_ORIENTATION

    # Position deviation (mm)
    dx = TARGET_POSITION[0]
    dy = TARGET_POSITION[1]
    dz = TARGET_POSITION[2]

    # Rotation deviation (degree)
    droll = TARGET_ORIENTATION[0]
    dpitch = TARGET_ORIENTATION[1]
    dyaw = TARGET_ORIENTATION[2]

    # Calculate linear velocity
    vx = max(min(dx * 2.5, MAX_VELO_X), -MAX_VELO_X)
    vy = max(min(dy * 2.5, MAX_VELO_Y), -MAX_VELO_Y)
    vz = max(min(dz * 2.5 - 40, MAX_VELO_Z), -MAX_VELO_Z)

    # Calculate angular velocity
    vroll = max(min(droll * 1.5, MAX_ANGULAR_VELO), -MAX_ANGULAR_VELO)
    vpitch = max(min(dpitch * 1.5, MAX_ANGULAR_VELO), -MAX_ANGULAR_VELO)
    vyaw = max(min(dyaw * 1.5, MAX_ANGULAR_VELO), -MAX_ANGULAR_VELO)

    CURRENT_VELOCITY = [vx, vy, vz, vroll, vpitch, vyaw]

    # Check whether is reached and execute grasp
    position_error = np.sqrt(dz ** 2)  # Only use height
    # orientation_error = np.sqrt(droll ** 2 + dpitch ** 2 + dyaw ** 2)

    if position_error < POSITION_TOLERANCE:
        print("reached, position_error:", position_error )
        execute_grasp_sequence(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                               h_model, epsilon, alpha, num_iter, perturbation, n_grasps, fig, axes)


def send_velocity_commands(arm):

    global CURRENT_VELOCITY

    if any(abs(v) > 0.1 for v in CURRENT_VELOCITY):
        try:
            arm.set_mode(5)
            arm.set_state(0)

            print("Servoing")
            arm.vc_set_cartesian_velocity([CURRENT_VELOCITY[0], CURRENT_VELOCITY[1], CURRENT_VELOCITY[2],
            0, 0, CURRENT_VELOCITY[5]], is_tool_coord=True, is_radian=False)

        except Exception as e:
            print(f"Failed to send velocitycommand: {e}")


def execute_grasp_sequence(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                               h_model, epsilon, alpha, num_iter, perturbation, n_grasps, fig, axes):

    global SERVO_ENABLED, GRIP_WIDTH

    SERVO_ENABLED = False
    print("Close servo and grasping...")

    arm.set_mode(0)
    arm.set_state(state=0)

    image_bundle = cam.get_image_bundle()
    rgb = image_bundle['rgb']
    depth = image_bundle['aligned_depth']

    try:
        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'],
                                                            pred['sin'], pred['width'])

            grasp_points = peak_local_max(q_img, min_distance=0, threshold_abs=0.1, num_peaks=1)

            if len(grasp_points) > 0:
                for grasp_point in grasp_points:
                    length = width_img[grasp_point[0], grasp_point[1]]
                    angle = ang_img[grasp_point[0], grasp_point[1]]
                    center = [grasp_point[0], grasp_point[1]]
                    data_array = [center[1] + 208, center[0] + 128]

                    aligned_depth_frame = image_bundle['aligned_depth_frame']
                    dis = aligned_depth_frame.get_distance(data_array[0], data_array[1])

                    if dis > 0:
                        x, y, g = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[data_array[0], data_array[1]],
                                                                  depth=dis)

                    if MIN_DEPTH < g < MAX_DEPTH:

                        smoothed_pos = pose_averager.update(np.array([x, y, g, angle]))

                        arm.set_gripper_position(length * 10, wait=True)

                        arm.set_tool_position(x=(smoothed_pos[1] + your_calib_x)*1000, y=(your_calib_y - smoothed_pos[0])*1000
                                              , z=0,
                                              roll=0, pitch=0, yaw=0,
                                              speed=50, is_radian=False)

                        #0.21
                        arm.set_tool_position(x=0, y=0
                                              , z=(smoothed_pos[2] - 0.17)*1000,
                                              roll=0, pitch=0, yaw=-smoothed_pos[3] * (180 / np.pi),
                                              speed=50, is_radian=False)

                        arm.set_gripper_position(0, wait=True)

                        arm.set_tool_position(x=0, y=0, z=-100, roll=0, pitch=0, yaw=0,
                                              speed=50, is_radian=False)

                        break

            # # Place object
            # place_object(arm)

            # Return to observation position
            return_to_observation_position(arm)

    except Exception as e:
        print(f"Failed to execute grasping: {e}")
        return_to_observation_position(arm)


def place_object(arm):

    global SERVO_ENABLED

    SERVO_ENABLED = True

    arm.set_mode(0)
    arm.set_state(state=0)

    arm.set_position(x=423.6, y=-314, z=300,
                     roll=-180, pitch=0, yaw=0, speed=100,
                     is_radian=False)
    arm.set_position(x=423.6, y=-314, z=167,
                     roll=-180, pitch=0, yaw=0, speed=100,
                     is_radian=False)

    arm.set_gripper_position(850, wait=True)


def return_to_observation_position(arm):

    global SERVO_ENABLED

    arm.set_mode(0)
    arm.set_state(state=0)

    arm.set_position(x=447.2, y=43, z=750,
                     roll=-180, pitch=0, yaw=0, speed=100,
                     is_radian=False)

    arm.set_gripper_position(850, wait=True)

    while True:
        ret, current_pose = arm.get_position()
        if ret == 0:
            tolerance = 3
            if (abs(current_pose[0] - 447.2) < tolerance and
                    abs(current_pose[1] - 43) < tolerance and
                    abs(current_pose[2] - 750) < tolerance):
                break
        time.sleep(0.1)

    SERVO_ENABLED = True

    print("Prepare next grasping...")


class Averager():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        return self.buffer.mean(axis=0)

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.buffer.mean(axis=0)

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.been_reset = True

def main():

    global SERVO_ENABLED, GRIP_WIDTH

    args = parse_args()

    # Initializing robot and gripper
    print("Initializing robot and gripper...")
    arm = initialize_robot()

    # Initializing camera
    print("Initializing camera...")
    cam = RealSenseCamera(device_id='your cameara ID')
    color_intrinsics = cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Initializing the compute device
    device = get_device(args.force_cpu)

    # Loading model
    print("Loading model...")
    net = torch.load(args.network)

    # Load hand segmentation model
    model_path = os.path.join(args.model_dir, args.model_name)
    h_model = DeepLabModel(model_path, 360, 360)

    # Load original patch
    logging.info('Loading patch...')
    perturbation = Image.open(args.pur_file)

    # Initializing filter
    pose_averager = Averager(4, 5)  # 5 frame
    logging.info('Initialize Done')

    # Get PQDG parameters
    epsion = args.epsilon
    alpha = args.alpha
    num_iter = args.num_iter

    # Get grasps number
    n_grasps = args.n_grasps

    # Initialize matplotlib figure ONCE
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    manager = fig.canvas.manager

    try:
        manager.window.wm_geometry("+50+50")  
    except:
        pass  

    # Begin servoing
    SERVO_ENABLED = True

    print("Open servo")
    print("Enter 'q' to quit")

    try:
        # Main control loop
        velocity_control_loop(arm, cam, cam_data, net, device, color_intrinsics, pose_averager,
                              h_model, epsion, alpha, num_iter, perturbation, n_grasps, fig, axes)

    except KeyboardInterrupt:
        print("Quit by user")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        print("Cleaning...")
        arm.reset(wait=True)
        cam.disconnect()
        print("Over")


if __name__ == '__main__':
    main()