import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
import time

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default ='epoch_48_iou_0.93',
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='E:/cornell_dataset/01/pcd0102r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='E:/cornell_dataset/01/pcd0102d.tiff',
                        help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=1,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load image
    logging.info('Loading image...')

    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)

    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)

        start_time = time.time()

        pred = net.predict(xc)
        end_time = time.time()
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        print(f"Total execution time: {end_time - start_time:.2f} seconds")

        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            None
            # fig.savefig('img_result.pdf')

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import argparse
# import logging
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import torch.utils.data
# from PIL import Image
# from hardware.device import get_device
# from inference.post_process import post_process_output
# from utils.data.camera_data import CameraData
# from utils.visualisation.plot import plot_results, save_results
#
# logging.basicConfig(level=logging.INFO)
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate network')
#     parser.add_argument('--network', type=str,default='deformable_epoch_40_iou_0.00', help='Path to saved network to evaluate')
#     parser.add_argument('--video_path', type=str, default='videos.mp4', help='Video file path')
#     parser.add_argument('--use-depth', type=int, default=0, help='Use Depth video for evaluation (1/0)')
#     parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB video for evaluation (1/0)')
#     parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per frame')
#     parser.add_argument('--save', type=int, default=0, help='Save the results')
#     parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
#                         help='Force code to run in CPU mode')
#     args = parser.parse_args()
#     return args
#
#
#
# def center_crop(frame, new_height, new_width):
#     height, width = frame.shape[:2]
#     start_x = (width - new_width) // 2
#     start_y = (height - new_height) // 2
#     return frame[start_y:start_y + new_height, start_x:start_x + new_width]
#
#
# def update(num, ax, video_reader, net, device, img_data, args):
#     full_frame = np.array(video_reader.get_data(num))
#     cropped_frame = center_crop(full_frame, 480, 640)  # 这里进行中心裁剪
#     x, _, _ = img_data.get_data(rgb=cropped_frame , depth=None)
#
#     with torch.no_grad():
#         xc = x.to(device)
#         pred = net.predict(xc)
#         q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
#
#         ax.clear()
#         plot_results(fig=ax.figure,
#                      rgb_img=img_data.get_rgb(cropped_frame, False),
#                      grasp_q_img=q_img,
#                      grasp_angle_img=ang_img,
#                      no_grasps=args.n_grasps,
#                      grasp_width_img=width_img)
#
# if __name__ == '__main__':
#     args = parse_args()
#
#     # Load Network
#     logging.info('Loading model...')
#     net = torch.load(args.network)
#     logging.info('Done')
#
#     # Get the compute device
#     device = get_device(args.force_cpu)
#     img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
#
#     # Load video
#     from imageio import get_reader
#
#     video_reader = get_reader(args.video_path)
#     try:
#         num_frames = video_reader.get_length()
#     except:
#         num_frames = 100  # Manually set the number of frames if get_length() doesn't work
#
#     if num_frames == float('inf'):
#         num_frames = 100  # Manually set the number of frames
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     ani = animation.FuncAnimation(fig, update, frames=int(num_frames),
#                                   fargs=(ax, video_reader, net, device, img_data, args))
#
#     if args.save:
#         ani.save('output_video.mp4', writer='ffmpeg', fps=30)
#
#     plt.show()
