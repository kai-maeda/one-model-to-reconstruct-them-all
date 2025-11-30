import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import glob
import random
import math
import argparse

from DL_train import TextureUNet, PatchTextureDataset, get_timestep_from_noise_level, add_noise_level, sample_from_partial, high_frequency_loss

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, help="the data key of the set. EX: STONE-ARCH for stone and architecture textures.")

parser.add_argument('-c', type=str, help="cache or not")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXTURE_DIRS = {"LOCALDEBUG": "/Users/slitf/Downloads/stone_masonry/", "NORMALS": "png_input/normals/",
                "PLANTS": "png_input/diffuse/organized_textures/nature_foliage/",
                "SNOW": "png_input/diffuse/organized_textures/snow_ice/",
                "ARCHIT": "png_input/diffuse/organized_textures/stone_masonry/",
                "TERRAIN": "png_input/diffuse/organized_textures/terrain_dirt/",
                "CLOTHING": "png_input/diffuse/organized_textures/armors/"}
TEXTURE_DIR = TEXTURE_DIRS[args.d]

SAVE_DIR = "diffusion_run/" + args.d + "/"
MODEL_PATH = os.path.join(SAVE_DIR, "model")
RESULTS_DIR = "results_test/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters for testing
PATCH_SIZE = 256
BATCH_SIZE = 4
MAX_TRAINING_NOISE_LEVEL = 0.20

# Load the trained model
def load_model(model_path):
    model = TextureUNet(in_ch=3, base_ch=128, time_emb_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Test dataset preparation
def prepare_test_dataset():
    # Assuming that you want to test patches from the same dataset format as training
    test_ds = PatchTextureDataset(
        root_dir=TEXTURE_DIR,
        patch_size=PATCH_SIZE,
        patches_per_image=1,  # Just 1 patch per image for testing
        max_images=10  # Change to your desired number of test images
    )
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

# Calculate PSNR, SSIM, LPIPS (You can add the relevant LPIPS and SSIM implementations here)
def calculate_metrics(model, dataloader):
    psnr_list, ssim_list, lpips_list = [], [], []
    for x0 in tqdm(dataloader, desc="Testing"):
        x0 = x0.to(DEVICE)

        # Compute timestep based on MAX_TRAINING_NOISE_LEVEL
        noise_level_tensor = torch.tensor(MAX_TRAINING_NOISE_LEVEL).to(DEVICE)
        k = get_timestep_from_noise_level(noise_level_tensor, alpha_hat)

        # Add noise to the original image (similar to training)
        xk_batch = add_noise_level(x0, noise_level_tensor)
        
        # Denoise
        x_denoised = sample_from_partial(model, xk_batch, k)

        # Calculate PSNR, SSIM, and LPIPS for the current batch
        # Example of PSNR calculation
        psnr = 10 * torch.log10(1.0 / F.mse_loss(x_denoised, x0))
        psnr_list.append(psnr.item())

        # TODO: You should implement or use the relevant LPIPS and SSIM libraries here

    return psnr_list, ssim_list, lpips_list

# Save comparison images
def save_comparisons(x0_batch, xk_batch, x_denoised, epoch):
    all_imgs = torch.cat([
        (x0_batch + 1) / 2,  # Original patches (clean)
        (xk_batch + 1) / 2,  # Degraded (noise)
        (x_denoised + 1) / 2  # Denoised result
    ], dim=0)

    save_image(all_imgs, os.path.join(RESULTS_DIR, f"test_comparison_epoch_{epoch}.png"), nrow=4)

# Main function for testing
def test():
    print("=" * 60)
    print("Starting testing...")
    print("=" * 60)

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Prepare the test dataset
    test_dl = prepare_test_dataset()

    # Testing loop
    psnr_list, ssim_list, lpips_list = calculate_metrics(model, test_dl)

    # Print average results
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)

    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")

    # Save some comparison images
    for i, (x0_batch, xk_batch, x_denoised) in enumerate(zip(test_dl, test_dl, test_dl)):
        save_comparisons(x0_batch, xk_batch, x_denoised, i)

    print("Testing completed!")

if __name__ == "__main__":
    test()