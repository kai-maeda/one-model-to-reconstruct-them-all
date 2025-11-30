import torch
import numpy as np
import lpips
import torchmetrics
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision.utils import save_image

def reconstruct_image_from_patches(patches, image_size, patch_size, patches_per_row):
    """
    Reconstruct the full image from patches. Assumes non-overlapping patches.
    
    patches: Tensor of shape [batch_size, C, patch_size, patch_size]
    image_size: (height, width) of the original image
    patch_size: the size of each patch
    patches_per_row: number of patches per row in the full image
    
    Returns: Reconstructed image
    """
    C, H, W = image_size
    rows = (H + patch_size - 1) // patch_size
    cols = (W + patch_size - 1) // patch_size
    
    # Initialize a blank image to hold the reassembled patches
    full_image = torch.zeros((C, H, W), dtype=patches.dtype)
    
    patch_idx = 0
    for i in range(rows):
        for j in range(cols):
            patch = patches[patch_idx]
            full_image[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch
            patch_idx += 1
    return full_image


# PSNR
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr_value = 10 * torch.log10(1.0 / mse)  # max value is assumed to be 1.0
    return psnr_value

# SSIM
def compute_ssim(img1, img2):
    # Convert to numpy for SSIM calculation
    img1 = img1.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C]
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1, img2, multichannel=True)

# LPIPS
lpips_fn = lpips.LPIPS(net='alex').cuda()  # Using AlexNet pretrained weights

def compute_lpips(img1, img2):
    return lpips_fn(img1, img2).item()

# Example of comparing images
def evaluate_metrics(model, test_loader, device):
    model.eval()
    psnr_values, ssim_values, lpips_values = [], [], []
    
    for inputs, targets in test_loader:  # Assuming test_loader gives batches of (inputs, targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reconstruct or predict the outputs
        with torch.no_grad():
            output = model(inputs)  # Assuming the model produces the output image
        
        # Reassemble the patches if necessary
        full_output = reconstruct_image_from_patches(output, image_size=(3, 256, 256), patch_size=PATCH_SIZE, patches_per_row=8)
        full_target = reconstruct_image_from_patches(targets, image_size=(3, 256, 256), patch_size=PATCH_SIZE, patches_per_row=8)
        
        # Compute metrics
        psnr_value = psnr(full_output, full_target)
        ssim_value = compute_ssim(full_output, full_target)
        lpips_value = compute_lpips(full_output, full_target)
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
    
    avg_psnr = torch.mean(torch.tensor(psnr_values))
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    print(f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')
