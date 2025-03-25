import warnings
from datetime import datetime
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results_og(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None,
        point=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps,point=point)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()

def plot_results(
        fig,
        no_grasps,
        rgb_img,
        grasp_q_img,
        grasp_q_img_1,
        grasp_angle_img,
        grasp_width_img,
        rgb_img_r,
        grasp_q_img_r,
        grasp_q_img_r1,
        grasp_angle_img_r,
        grasp_width_img_r,
        point=None):

    # Get grasps
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img,
                       no_grasps=no_grasps, point=point)
    gs_1 = detect_grasps(grasp_q_img_1, grasp_angle_img, width_img=grasp_width_img,
                         no_grasps=no_grasps, point=point)
    gs_r = detect_grasps(grasp_q_img_r, grasp_angle_img_r, width_img=grasp_width_img_r,
                         no_grasps=no_grasps, point=point)
    gs_r1 = detect_grasps(grasp_q_img_r1, grasp_angle_img_r, width_img=grasp_width_img_r,
                          no_grasps=no_grasps, point=point)

    # First row
    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(rgb_img)
    for g in gs:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=3)
        ax.add_patch(circle)
    ax.set_title('Original Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(rgb_img)
    for g in gs_1:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=3)
        ax.add_patch(circle)
    ax.set_title('Original-SZ Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(rgb_img_r)
    for g in gs_r:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=3)
        ax.add_patch(circle)
    ax.set_title('QFAAP-NSZ Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(rgb_img_r)
    for g in gs_r1:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=3)
        ax.add_patch(circle)

    ax.set_title('QFAAP Grasp')
    ax.axis('off')


    # Second row
    ax = fig.add_subplot(2, 4, 5)
    plot = ax.imshow(grasp_q_img, cmap='hot', vmin=0, vmax=1)
    for g in gs:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    ax.set_title('Original Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(2, 4, 6)
    plot = ax.imshow(grasp_q_img_1, cmap='hot', vmin=0, vmax=1)
    for g in gs_1:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    ax.set_title('Original-SZ Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(2, 4, 7)
    plot = ax.imshow(grasp_q_img_r, cmap='hot', vmin=0, vmax=1)
    for g in gs_r:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    ax.set_title('QFAAP-NSZ Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(2, 4, 8)
    plot = ax.imshow(grasp_q_img_r1, cmap='hot', vmin=0, vmax=1)
    for g in gs_r1:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=6)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    ax.set_title('QFAAP Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.pause(0.1)
    fig.canvas.draw()


def save_results(
        no_grasps,
        rgb_img,
        grasp_q_img,
        grasp_q_img_1,
        grasp_angle_img,
        grasp_width_img,
        rgb_img_r,
        grasp_q_img_r,
        grasp_q_img_r1,
        grasp_angle_img_r,
        grasp_width_img_r,
        point=None):

    # Get grasps
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img,
                       no_grasps=no_grasps, point=point)
    gs_1 = detect_grasps(grasp_q_img_1, grasp_angle_img, width_img=grasp_width_img,
                         no_grasps=no_grasps, point=point)
    gs_r = detect_grasps(grasp_q_img_r, grasp_angle_img_r, width_img=grasp_width_img_r,
                         no_grasps=no_grasps, point=point)
    gs_r1 = detect_grasps(grasp_q_img_r1, grasp_angle_img_r, width_img=grasp_width_img_r,
                          no_grasps=no_grasps, point=point)

    # First row
    plt.ion()
    plt.clf()
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('Original Grasp')
    ax.axis('off')
    fig.savefig('results/Original Grasp.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs_1:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('Original-SZ Grasp')
    ax.axis('off')
    fig.savefig('results/Original-SZ Grasp.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img_r)
    for g in gs_r:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('QFAAP-NSZ Grasp')
    ax.axis('off')
    fig.savefig('results/QFAAP-NSZ Grasp.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img_r)
    for g in gs_r1:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('QFAAP Grasp')
    ax.axis('off')
    fig.savefig('results/QFAAP Grasp.png', bbox_inches='tight', pad_inches=0)


    # Second row
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='hot', vmin=0, vmax=1)
    for g in gs:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)
    # ax.set_title('Original Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/Original Quality.png', bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img_1, cmap='hot', vmin=0, vmax=1)
    for g in gs_1:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)
    # ax.set_title('Original-SZ Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/Original-SZ Quality.png', bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img_r, cmap='hot', vmin=0, vmax=1)
    for g in gs_r:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)

    # ax.set_title('QFAAP-NSZ Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/QFAAP-NSZ Quality.png', bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img_r1, cmap='hot', vmin=0, vmax=1)
    for g in gs_r1:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)
    # ax.set_title('QFAAP Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/QFAAP Quality.png', bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.pause(0.1)
    fig.canvas.draw()


def save_results2(
        no_grasps,
        rgb_img,
        rgb_img_r,
        grasp_q_img,
        grasp_angle_img,
        grasp_width_img,
        grasp_q_img_r,
        grasp_angle_img_r,
        grasp_width_img_r,
        point=None):
    # Get grasps
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img,
                       no_grasps=no_grasps, point=point)
    gs_r = detect_grasps(grasp_q_img_r, grasp_angle_img_r, width_img=grasp_width_img_r,
                         no_grasps=no_grasps, point=point)

    # First row
    plt.ion()
    plt.clf()
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('Original Grasp')
    ax.axis('off')
    fig.savefig('results/AQP Grasp.png', bbox_inches='tight', pad_inches=0)

    plt.ion()
    plt.clf()
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    ax.imshow(rgb_img_r)
    for g in gs_r:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='lime', facecolor='none', linewidth=6)
        ax.add_patch(circle)
    # ax.set_title('Original Grasp')
    ax.axis('off')
    fig.savefig('results/(AQP+PQGD) Grasp.png', bbox_inches='tight', pad_inches=0)

    # Second row
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='hot', vmin=0, vmax=1)
    for g in gs:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)
    # ax.set_title('Original Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/AQP Quality.png', bbox_inches='tight', pad_inches=0)


    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img_r, cmap='hot', vmin=0, vmax=1)
    for g in gs_r:
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=20)
        radius = 10
        circle = Circle((g.center[1], g.center[0]), radius, edgecolor='white', facecolor='none', linewidth=6, alpha=0.5)
        ax.add_patch(circle)

    # ax.set_title('QFAAP-NSZ Quality')
    ax.axis('off')
    # cbar = plt.colorbar(plot, ax=ax, orientation='vertical')
    # cbar.ax.tick_params(labelsize=8)
    fig.savefig('results/(AQP+PQGD) Quality.png', bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.pause(0.1)
    fig.canvas.draw()