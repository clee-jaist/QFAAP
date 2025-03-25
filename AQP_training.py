import argparse
import logging
import datetime
import json
import os
import torch
import numpy as np
import torch.optim as optim
import random
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from hardware.device import get_device
from utils.data import get_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train adversarial patch')

    # Dataset and Model configurations
    parser.add_argument('--network', type=str, default='QFAAP_models/Jacquard_TFgrasp_epoch_40_iou_0.81', help='Path to saved network')
    parser.add_argument('--dataset', type=str, default='jacquard', help='Dataset name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='B:/rename_jacquard', help='Path to dataset')
    parser.add_argument('--input-size', type=int, default=224, help='Input image size, 300(ggcnn)/224(others)')
    parser.add_argument('--split', type=float, default=0.9, help='Train-test split ratio')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB images')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth images (1/0)')
    parser.add_argument('--ds-shuffle', action='store_true', default=True, help='Shuffle the dataset')

    # AQP parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patch-size', type=int, default=224, help='Patch size, 300(ggcnn)/224(others)')
    parser.add_argument('--output-path', type=str, default='adversarial_patch_results/', help='Output directory')

    # PQGD parameters
    parser.add_argument('--epsilon', type=float, default=8/255, help='Maximum perturbation')
    parser.add_argument('--alpha', type=float, default=0.008, help='Step size for perturbation')
    parser.add_argument('--num-iter', type=int, default=1, help='Number of iterations for perturbation')

    # Log and Device configurations
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False, help='Force to use CPU')
    parser.add_argument('--logdir', type=str, default='adv_logs/', help='Log directory')

    args = parser.parse_args()
    return args


def save_validation_results(Avg_AQP_ratios, Avg_PQGD_ratios, Avg_AQP_predict_time,
                            Avg_PQGD_predict_time, log_dir, epoch):

    results_file = os.path.join(log_dir, "validation_results.txt")

    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("Epoch, Avg_AQP_ratios, Avg_PQGD_ratios, Avg_AQP_predict_time, Avg_PQGD_predict_time\n")

    with open(results_file, 'a') as f:
        f.write(f"{epoch + 1}, {Avg_AQP_ratios:.4f}, {Avg_PQGD_ratios:.4f}, "
                f"{Avg_AQP_predict_time:.4f}, {Avg_PQGD_predict_time:.4f}\n")


def save_patch_results(patch, epoch, Avg_AQP_ratios, Avg_PQGD_ratios, output_dir='output_results'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    patch = patch.squeeze().cpu().detach().numpy()
    patch = np.moveaxis(patch, 0, -1)
    patch = np.clip(patch, 0, 1)

    patch_image = Image.fromarray((patch * 255).astype(np.uint8))
    patch_image.save(os.path.join(output_dir, f'epoch_{epoch}_AQP_{Avg_AQP_ratios:.4f}'
                                              f'_PQGD_{Avg_PQGD_ratios:.4f}.png'))


def normalize_positions(pos_tensor):

    normalized_pos = torch.zeros_like(pos_tensor)

    for i in range(pos_tensor.size(0)):
        pos_image = pos_tensor[i]
        pos_min = pos_image.min()
        pos_max = pos_image.max()
        normalized_pos[i] = (pos_image - pos_min) / (pos_max - pos_min)

    normalized_pos = torch.clamp(normalized_pos, 0, 1)

    return normalized_pos


def process_patch_predictions(model, patched_images, patch_mask):

    outputs = model.predict(patched_images)
    pos = normalize_positions(outputs['pos'])

    pos_patch_values = pos * patch_mask
    count_above_0_5 = (pos_patch_values > 0.5).sum(dim=(2, 3))
    mask_pixel_count = patch_mask.sum(dim=(2, 3))

    ratio = count_above_0_5.float() / mask_pixel_count.float()
    average_ratio = ratio.mean().item()

    return average_ratio


def total_variation(adv_patch):

    # x
    tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=2)
    tvcomp1 = torch.sum(tvcomp1, dim=1)

    # y
    tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=2)
    tvcomp2 = torch.sum(tvcomp2, dim=1)

    tv = tvcomp1 + tvcomp2
    tv = tv.sum()

    return tv / torch.numel(adv_patch)


def rescale_patch(patch, min_scale=0.5, max_scale=3, scale = None):

    if scale:
        scale_factor = scale
    else:
        scale_factor = random.uniform(min_scale, max_scale)

    new_size = int(patch.size(1) * scale_factor), int(patch.size(2) * scale_factor)
    rescaled_patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)
    return rescaled_patch.squeeze(0), scale_factor


def apply_patch(image, patch, input_size, min_scale=0.5, max_scale=1.5):

    batch_size = image.size(0)

    patch, scale_factor = rescale_patch(patch, min_scale, max_scale, scale = None)
    rescaled_patch_size = patch.size(1)

    if input_size <= rescaled_patch_size:
        x_offset = np.zeros(batch_size, dtype=int)
        y_offset = np.zeros(batch_size, dtype=int)
    else:
        x_offset = np.random.randint(0, input_size - rescaled_patch_size, size=batch_size)
        y_offset = np.random.randint(0, input_size - rescaled_patch_size, size=batch_size)

    patched_images = image.clone()
    patch_mask = torch.zeros_like(image, device=image.device)

    for i in range(batch_size):
        patched_images[i, :, x_offset[i]:x_offset[i] + rescaled_patch_size, y_offset[i]:y_offset[i] + rescaled_patch_size] = patch
        patched_images.data.clamp_(0, 1)
        patch_mask[i, :, x_offset[i]:x_offset[i] + rescaled_patch_size, y_offset[i]:y_offset[i] + rescaled_patch_size] = 1.0

    return x_offset, y_offset, patched_images, patch_mask


def pqgd(model, inputs, epsilon, alpha, num_iter, mask, ori_img, device):

    perturbed_inputs = inputs.clone().detach().to(device)
    mask = mask.to(device)

    perturbed_inputs.requires_grad_(True)

    for _ in range(num_iter):

        with torch.enable_grad():
            model.eval()
            model.zero_grad()

            # Loss
            outputs = model.predict(perturbed_inputs)
            pos = normalize_positions(outputs['pos'])
            pos_patch_values = pos * mask
            pos_mean = torch.mean(pos_patch_values)
            loss = -pos_mean

        loss.backward()

        # Compute perturbation
        adv = perturbed_inputs - alpha * perturbed_inputs.grad.sign()
        perturbation = torch.clamp(adv - ori_img, min=-epsilon, max=epsilon)

        # Apply perturbation only to the masked region
        if perturbed_inputs.shape == torch.Size([args.batch_size, 3, 224, 224]):
            mask = mask.expand(args.batch_size, 3, 224, 224)
        else:
            mask = mask.expand(1, 3, 224, 224)

        min = ori_img.min()
        max = ori_img.max()
        perturbed_inputs = ori_img.clone()
        perturbed_inputs[mask == 1] = torch.clamp(
            ori_img + perturbation, min=min, max=max
        )[mask == 1]

        # Detach and re-enable gradient tracking
        perturbed_inputs = perturbed_inputs.detach_()
        perturbed_inputs.requires_grad_(True)

    return perturbed_inputs

def train_adversarial_patch(args):

    # Set up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.network.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

    # Loading dataset
    logging.info('Loading dataset...')

    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path, output_size=args.input_size, random_rotate=True, random_zoom=True,
                      include_rgb=args.use_rgb, include_depth=args.use_depth)

    indices = list(range(dataset.length))
    split = int(np.floor(args.split * len(indices)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices))

    # Loading model
    logging.info('Loading model...')
    model = torch.load(args.network)
    device = get_device(args.force_cpu)
    model.to(device)

    # Initializing patch
    patch = torch.rand((3, args.patch_size, args.patch_size), device=device)
    patch.requires_grad_(True)
    optimizer = optim.Adam([patch], lr=0.03, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # Training
    logging.info('Training...')
    for epoch in range(args.epochs):
        model.eval()
        model.zero_grad()
        logging.info(f'Epoch {epoch + 1}/{args.epochs}')
        epoch_start_time = time.time()

        epoch_patch_loss = 0.0
        epoch_dif_loss = 0.0
        epoch_tv_loss = 0.0
        epoch_total_loss = 0.0
        batch_idx = 0

        while batch_idx <= args.batches_per_epoch:
            for idx, (images, _, _, _, _) in tqdm(enumerate(train_loader), total=args.batches_per_epoch):

                # Use batches per epoch to make training on different sized datasets more equivalent.
                batch_idx += 1
                if batch_idx >= args.batches_per_epoch:
                    break

                # Apply patch
                images = images.to(device)
                x_offset, y_offset, patched_images, patch_mask = apply_patch(images, patch, args.input_size,
                                                                             min_scale=0.1, max_scale=1)

                inverse_patch_mask = np.logical_not(patch_mask.cpu().numpy()).astype(int)
                inverse_patch_mask = torch.tensor(inverse_patch_mask).cuda()

                patch_mask = patch_mask[:, :1, :, :]  # shape: batch_size *1 * 224 *224
                inverse_patch_mask = inverse_patch_mask[:, :1, :, :]

                outputs = model.predict(patched_images)

                # Normalizing outputs
                pos = normalize_positions(outputs['pos'])
                patch_quality = pos * patch_mask

                #===============================================================================================
                # Patch loss

                min_values = patch_quality.min(dim=3).values.min(dim=2).values # shape: batch_size * 1 * 224 * 224
                var_values = patch_quality.var(dim=(2, 3), unbiased=False)
                patch_var = var_values.mean().item()
                patch_mean = torch.mean(patch_quality)
                patch_loss = - patch_mean + 0.1 * patch_var

                # Difference loss
                image_quality = pos * inverse_patch_mask
                max_values = image_quality.max(dim=3).values.max(dim=2).values
                dif = torch.mean(abs(min_values - max_values)) # simple l1 loss

                # Tv loss
                tv_loss = total_variation(patch)

                # Total loss
                loss = patch_loss + 0.1 * dif + 0.5 * torch.max(tv_loss, torch.tensor(0.1).cuda())
                # ===============================================================================================

                # Update cumulative losses
                epoch_patch_loss += patch_loss
                epoch_dif_loss += dif
                epoch_tv_loss += tv_loss
                epoch_total_loss += loss

                # Compute gradients
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                patch.data.clamp_(0, 1)

        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")

        # Timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Logging batch average losses for the current epoch
        logging.info(f'Epoch {epoch + 1} Patch_Loss: {epoch_patch_loss / (args.batches_per_epoch*args.batch_size):.4f}')
        logging.info(f'Epoch {epoch + 1} Dif_Loss: {(epoch_dif_loss * 0.1) / (args.batches_per_epoch*args.batch_size):.4f}')
        logging.info(f'Epoch {epoch + 1} Tv_Loss: {(epoch_tv_loss * 0.5) / (args.batches_per_epoch*args.batch_size):.4f}')
        logging.info(f'Epoch {epoch + 1} Total Loss: {epoch_total_loss / (args.batches_per_epoch*args.batch_size):.4f}')
        logging.info(f'Epoch {epoch + 1} Training Time: {epoch_duration:.2f} seconds')

        # Testing
        AQP_ratios = []
        PQGD_ratios = []
        AQP_time = []
        PQGD_time = []

        with torch.no_grad():
            logging.info('Validating...')
            for images, _, _, _, _ in val_loader:
                images = images.to(device)

                x_offset, y_offset, patched_images, patch_mask = apply_patch(images, patch,
                                                args.input_size, min_scale=0.1, max_scale=1)

                patch_mask = patch_mask[:, :1, :, :]

                # AQP Testing
                AQP_start_time = time.time()

                average_ratio = process_patch_predictions(model, patched_images, patch_mask) # Predicting

                AQP_end_time = time.time()
                AQP_duration = AQP_end_time - AQP_start_time
                AQP_time.append(AQP_duration)
                AQP_ratios.append(average_ratio)

                # PQGD Testing
                ori_img = patched_images.data
                PQGD_start_time = time.time()

                PQGD_images = pqgd(model, patched_images, args.epsilon, args.alpha,
                                   args.num_iter, patch_mask, ori_img, device)      # PQGD iteration
                average_ratio = process_patch_predictions(model, PQGD_images, patch_mask)    # Predicting

                PQGD_end_time = time.time()
                PQGD_duration = PQGD_end_time - PQGD_start_time
                PQGD_time.append(PQGD_duration)
                PQGD_ratios.append(average_ratio)

            Avg_AQP_ratios = np.mean(AQP_ratios)
            Avg_PQGD_ratios = np.mean(PQGD_ratios)
            Avg_AQP_predict_time = np.sum(AQP_time) / len(val_loader)
            Avg_PQGD_prediect_time = np.sum(PQGD_time) / len(val_loader)

            logging.info(f'Epoch {epoch + 1} Avg_AQP_ratios: {Avg_AQP_ratios:.4f}')
            logging.info(f'Epoch {epoch + 1} Avg_PQGD_ratios: {Avg_PQGD_ratios:.4f}')
            logging.info(f'Epoch {epoch + 1} Avg_AQP_predict_time: {Avg_AQP_predict_time:.4f} seconds')
            logging.info(f'Epoch {epoch + 1} Avg_PQGD_prediect_time: {Avg_PQGD_prediect_time:.4f} seconds')

            # Save validation results
            save_validation_results(Avg_AQP_ratios, Avg_PQGD_ratios, Avg_AQP_predict_time,
                                    Avg_PQGD_prediect_time, save_folder, epoch)

        # Save patch results
        save_patch_results(patch, epoch + 1, Avg_AQP_ratios, Avg_PQGD_ratios, save_folder)

    logging.info('Training completed')


if __name__ == '__main__':
    args = parse_args()
    train_adversarial_patch(args)