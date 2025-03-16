import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import random
from tqdm import tqdm

class SimpleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        pool = torch.mean(x, dim=[2, 3], keepdim=True)
        attention = self.sigmoid(self.conv(pool))
        return x * attention

# Lightweight EfficientFormer-inspired feature extractor
class LightweightFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[48, 96, 192]):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1, groups=embed_dims[0]),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, padding=1, groups=embed_dims[1]),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU(),
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
    
    def forward(self, x):
        # Feature hierarchy for multi-scale features
        features = []
        
        x = self.patch_embed(x)
        features.append(x)
        
        identity = x
        x = self.stage1(x)
        if identity.shape[1] == x.shape[1]:
            x = x + identity
        features.append(x)
        
        x = self.down1(x)
        x = self.stage2(x)
        features.append(x)
        
        return features

# Sparse depth estimator with physics-guided constraints
class SparseDepthEstimator(nn.Module):
    def __init__(self, in_channels=128, key_points=512):
        super().__init__()
        self.key_points = key_points
        
        # This layer will predict sparse depth points and their confidence
        self.sparse_depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1)  # 1 channel for depth, 1 for confidence
        )

        self.attention = SimpleAttention(in_channels)
    
    def forward(self, features):
        last_feature = self.attention(features[-1])
        sparse_output = self.sparse_depth_head(last_feature)
        
        # Split into depth and confidence
        depth_pred = sparse_output[:, 0:1]
        confidence = sparse_output[:, 1:2]
        
        # Apply softmax on confidence to get confidence weighting
        confidence = torch.softmax(confidence.view(confidence.shape[0], -1), dim=1)
        confidence = confidence.view_as(depth_pred)
        confidence = F.avg_pool2d(confidence, kernel_size=3, stride=1, padding=1)
        
        # Get top-k confident points
        batch_size = depth_pred.shape[0]
        h, w = depth_pred.shape[2], depth_pred.shape[3]
        
        confidence_flat = confidence.view(batch_size, -1)
        depth_flat = depth_pred.view(batch_size, -1)
        
        # Get top-k keypoints based on confidence
        _, indices = torch.topk(confidence_flat, k=min(self.key_points, h*w), dim=1)
        
        # Extract sparse depths at those points
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, self.key_points)
        sparse_depths = depth_flat[batch_indices, indices]
        
        # Convert flat indices to 2D coordinates
        y_indices = (indices // w).float() / h
        x_indices = (indices % w).float() / w
        
        # Stack coordinates and depth values
        sparse_points = torch.stack([
            x_indices,  # x coord (normalized 0-1)
            y_indices,  # y coord (normalized 0-1)
            sparse_depths  # depth value
        ], dim=2)
        
        return sparse_points, depth_pred, confidence

# Lightweight depth completion module
class DepthCompletion(nn.Module):
    def __init__(self, in_channels=[32, 64, 128], out_channels=1):
        super().__init__()
        
        # Decoder path with skip connections (lightweight U-Net style)
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], kernel_size=1),
            nn.BatchNorm2d(in_channels[1]),
            nn.GELU()
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels[1]*2, in_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[1]),
            nn.GELU()
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], kernel_size=1),
            nn.BatchNorm2d(in_channels[0]),
            nn.GELU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels[0]*2, in_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[0]),
            nn.GELU()
        )
        
        # Final depth prediction layer
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=3, padding=1),
            nn.ReLU()  # Ensure positive depth values
        )
    
    def forward(self, features, sparse_depth=None):
        # Decoder with skip connections
        x = self.up1(features[2])
        x = F.interpolate(x, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, features[1]], dim=1)
        x = self.decoder1(x)
        
        x = self.up2(x)
        x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, features[0]], dim=1)
        x = self.decoder2(x)
        
        # Make final depth prediction
        depth = self.depth_head(x)
        
        return depth

# Complete model combining all components
class PhysicsGuidedDepthEstimation(nn.Module):
    def __init__(self, img_height=192, img_width=640):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width

        embed_dims = [48, 96, 128]
        
        # Components
        self.feature_extractor = LightweightFeatureExtractor(embed_dims=embed_dims)
        self.sparse_estimator = SparseDepthEstimator(in_channels=embed_dims[-1])
        self.depth_completion = DepthCompletion(in_channels=embed_dims)
        
        # Camera parameters (these would be calibrated in practice)
        self.register_buffer('focal_length', torch.tensor([718.856]))
        self.register_buffer('baseline', torch.tensor([0.5]))
    
    def thin_lens_physics_prior(self, depth_pred):
        """Apply thin lens equation as a physics prior to refine depth"""
        # Filter out extreme values
        valid_depth = torch.clamp(depth_pred, min=0.1, max=10.0)
        
        # Apply gentler weighting to avoid overcorrection
        alpha = 0.7  # Reduce weight of physics correction (was 0.7)
        
        # Simulated disparity (inverse of depth)
        disparity = 1.0 / (valid_depth + 1e-6)
        
        # Apply thin lens equation: d = f*B/D
        physics_depth = self.focal_length * self.baseline * disparity
        
        # Blend original prediction with physics-based correction
        refined_depth = alpha * valid_depth + (1 - alpha) * physics_depth
        
        return refined_depth
    
    def forward(self, x):
        # Get input image dimensions
        _, _, H, W = x.shape
        
        # Extract multi-scale features
        features = self.feature_extractor(x)
        
        # Sparse depth estimation
        sparse_points, sparse_depth, confidence = self.sparse_estimator(features)
        
        # Complete depth map
        dense_depth = self.depth_completion(features)
        
        # Resize to original input resolution
        dense_depth = F.interpolate(dense_depth, size=(H, W), mode='bilinear', align_corners=False)
        
        # Apply physics-based correction
        refined_depth = self.thin_lens_physics_prior(dense_depth)
        
        return {
            'depth': refined_depth,
            'sparse_points': sparse_points,
            'confidence': confidence
        }

# Physics-guided losses
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for depth loss
        self.beta = beta    # Weight for smoothness loss
        self.gamma = gamma  # Weight for physics consistency loss
    
    def depth_loss(self, pred, target, mask=None):
        """Compute depth loss with scale-invariant option"""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        
        # Absolute relative error
        abs_rel = torch.mean(torch.abs(pred - target) / (target + 1e-6))
        
        # Scale-invariant MSE loss
        diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        si_mse = torch.mean(diff * diff) - 0.5 * torch.mean(diff) ** 2
        
        return abs_rel + si_mse
    
    def edge_aware_smoothness_loss(self, depth, image):
        """Edge-aware smoothness loss to preserve edges"""
        depth_grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        image_grad_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        image_grad_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
        
        weights_x = torch.exp(-image_grad_x)
        weights_y = torch.exp(-image_grad_y)
        
        smoothness_x = depth_grad_x * weights_x
        smoothness_y = depth_grad_y * weights_y
        
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)
    
    def physics_consistency_loss(self, depth):
        """Ensure physics consistency - ground should be roughly planar"""
        # Assuming lower half of image is mostly ground
        h = depth.shape[2]
        ground_region = depth[:, :, h//2:, :]
        
        # Compute gradients in ground region
        ground_grad_x = torch.abs(ground_region[:, :, :, :-1] - ground_region[:, :, :, 1:])
        ground_grad_y = torch.abs(ground_region[:, :, :-1, :] - ground_region[:, :, 1:, :])
        
        # Ground should be relatively smooth in depth
        return torch.mean(ground_grad_x) + torch.mean(ground_grad_y)
    
    def forward(self, outputs, targets, images):
        pred_depth = outputs['depth']
        target_depth = targets['depth']
        
        # Compute losses
        d_loss = self.depth_loss(pred_depth, target_depth)
        s_loss = self.edge_aware_smoothness_loss(pred_depth, images)
        p_loss = self.physics_consistency_loss(pred_depth)
        
        # Combine losses
        total_loss = self.alpha * d_loss + self.beta * s_loss + self.gamma * p_loss
        
        return total_loss, {
            'depth_loss': d_loss.item(),
            'smoothness_loss': s_loss.item(),
            'physics_loss': p_loss.item()
        }

class NYUv2BookstoreDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, height=192, width=256):
        """
        NYU Depth V2 Bookstore dataset loader
        
        Args:
            base_dir: Path to the bookstore dataset directory
            split: 'train' or 'val'
            transform: Optional transform to apply
            height, width: Target dimensions for resizing
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.height = height
        self.width = width
        
        # Get all PNG files (RGB images)
        self.rgb_files = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]
        self.rgb_files.sort()  # Ensure consistent ordering
        
        # Create corresponding depth file names
        self.depth_files = [f.replace('.jpg', '.png') for f in self.rgb_files]
        
        # Verify that all depth files exist
        for df in self.depth_files:
            if not os.path.exists(os.path.join(base_dir, df)):
                print(f"Warning: Depth file {df} not found")
        
        # Create train/val split (80/20)
        total_files = len(self.rgb_files)
        split_idx = int(total_files * 0.8)
        
        if split == 'train':
            self.rgb_files = self.rgb_files[:split_idx]
            self.depth_files = self.depth_files[:split_idx]
        else:  # validation
            self.rgb_files = self.rgb_files[split_idx:]
            self.depth_files = self.depth_files[split_idx:]
        
        print(f"Loaded {len(self.rgb_files)} {split} samples from {base_dir}")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Load RGB image (PNG)
        rgb_path = os.path.join(self.base_dir, self.rgb_files[idx])
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth image (JPG, grayscale)
        depth_path = os.path.join(self.base_dir, self.depth_files[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize images
        image = cv2.resize(image, (self.width, self.height))
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Normalize RGB to [0,1]
        image = image.astype(np.float32) / 255.0
        
        # For NYUv2, depth values might need to be scaled appropriately
        # The scale factor depends on how the depth was saved
        # According to typical NYUv2 processing, we assume depth is in millimeters
        # and we convert to meters (scale by 0.001)
        # depth = depth.astype(np.float32) / 255.0  # First normalize from [0,255]
        # depth = depth * 10.0  # Scale to realistic depth range (adjust as needed)

        # # CORRECTED: Proper NYUv2 depth scaling (typical values are in meters)
        depth = depth.astype(np.float32) / 1000.0  # Convert from millimeters to meters
        # # Remove extreme outliers
        depth = np.clip(depth, 0.1, 10.0)  # Clip to reasonable depth range (0.1m to 10m)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        
        sample = {
            'image': image,
            'depth': depth
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Function to generate depth maps for test images
def generate_depth_maps(model, test_images_dir, output_dir, img_height=192, img_width=256):
    """
    Generate depth maps for a directory of test images
    
    Args:
        model: Trained depth estimation model
        test_images_dir: Directory containing test images
        output_dir: Directory to save depth maps
        img_height, img_width: Height and width for resizing images
    """
    # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(test_images_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} test images")
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(test_images_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original dimensions for later resizing
        original_h, original_w = image.shape[:2]
        
        # Preprocess image: Resize, Normalize, Convert to Tensor
        image_resized = cv2.resize(image, (img_width, img_height))
        image_tensor = torch.from_numpy(image_resized).float() / 255.0  # Normalize to [0,1]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
        image_tensor = image_tensor.to(device)  # Move to GPU if available

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Ensure correct output format
        if isinstance(outputs, dict):
            depth_map = outputs['depth'].squeeze().cpu().numpy()  # Some models return dict output
        else:
            depth_map = outputs.squeeze().cpu().numpy()  # If model returns direct tensor
        
        # If model outputs log-depth, convert back to linear scale
        if np.min(depth_map) < 0:
            depth_map = np.exp(depth_map)

        # Resize depth map to original image dimensions
        depth_map_resized = cv2.resize(depth_map, (original_w, original_h))

        # Normalize depth map for visualization
        depth_colored = visualize_depth(depth_map_resized)

        # Save results
        base_name = os.path.splitext(img_file)[0]
        
        # Save raw depth data (numpy format)
        np.save(os.path.join(output_dir, f"{base_name}_depth.npy"), depth_map_resized)
        
        # Save colored visualization
        plt.imsave(os.path.join(output_dir, f"{base_name}_depth_colored.png"), depth_colored)
        
        # Create side-by-side visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Depth Map")
        plt.imshow(depth_colored)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"))
        plt.close()
        
        print(f"Processed {img_file}")

def visualize_depth(depth_map, cmap='plasma'):
    # Avoid invalid values
    min_depth, max_depth = np.percentile(depth_map, [2, 98])  # Robust scaling
    depth_norm = np.clip(depth_map, min_depth, max_depth)
    depth_norm = (depth_norm - min_depth) / (max_depth - min_depth + 1e-8)  # Normalize

    return plt.cm.get_cmap(cmap)(depth_norm)[..., :3]  # Remove alpha channel

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=30, device='cuda', save_dir='checkpoints'):
    """
    Train the depth estimation model
    
    Args:
        model: The PhysicsGuidedDepthEstimation model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        criterion: Loss function (PhysicsGuidedLoss)
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        save_dir: Directory to save model checkpoints
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Move model to device
    model = model.to(device)

    visualize_every = 200  # Visualize every 200 batches
    val_every = 500        # Validate every 500 batches
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {'depth_loss': 0.0, 'smoothness_loss': 0.0, 'physics_loss': 0.0}
        
        # Wrap dataloader in tqdm for progress bar
        for batch_idx, sample in enumerate(tqdm(train_loader)):
            # Move data to device
            images = sample['image'].to(device)
            target_depth = sample['depth'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            targets = {'depth': target_depth}
            loss, loss_components = criterion(outputs, targets, images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            for k, v in loss_components.items():
                train_loss_components[k] += v

            global_step += 1
            
            # Periodic validation
            if global_step % val_every == 0:
                model.eval()
                batch_val_loss = 0.0
                with torch.no_grad():
                    val_sample = next(iter(val_loader))
                    val_images = val_sample['image'].to(device)
                    val_depth = val_sample['depth'].to(device)
                    
                    val_outputs = model(val_images)
                    batch_val_loss, _ = criterion(val_outputs, {'depth': val_depth}, val_images)
                    
                    print(f"Step {global_step}: Val loss: {batch_val_loss.item():.4f}")
                
                model.train()

            # Periodic visualization
            if global_step % visualize_every == 0:
                model.eval()
                with torch.no_grad():
                    val_sample = next(iter(val_loader))
                    val_images = val_sample['image'].to(device)
                    val_depth = val_sample['depth'].to(device)
                    
                    val_outputs = model(val_images)
                    
                    # Save a visualization
                    os.makedirs(os.path.join(save_dir, 'progress'), exist_ok=True)
                    vis_img = val_images[0].cpu().numpy().transpose(1, 2, 0)
                    vis_gt = val_depth[0, 0].cpu().numpy()
                    vis_pred = val_outputs['depth'][0, 0].cpu().numpy()
                    
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.title("RGB Input")
                    plt.imshow(vis_img)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.title("Ground Truth")
                    plt.imshow(visualize_depth(vis_gt))
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.title("Prediction")
                    plt.imshow(visualize_depth(vis_pred))
                    plt.axis('off')
                    
                    plt.savefig(os.path.join(save_dir, 'progress', f'step_{global_step}.png'))
                    plt.close()
                
                model.train()
        
        # Average training losses
        train_loss /= len(train_loader)
        for k in train_loss_components:
            train_loss_components[k] /= len(train_loader)
        
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'depth_loss': 0.0, 'smoothness_loss': 0.0, 'physics_loss': 0.0}
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(val_loader)):
                # Move data to device
                images = sample['image'].to(device)
                target_depth = sample['depth'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                targets = {'depth': target_depth}
                loss, loss_components = criterion(outputs, targets, images)
                
                # Track losses
                val_loss += loss.item()
                for k, v in loss_components.items():
                    val_loss_components[k] += v
        
        # Average validation losses
        val_loss /= len(val_loader)
        for k in val_loss_components:
            val_loss_components[k] /= len(val_loader)
        
        val_losses.append(val_loss)
        
        # Call scheduler.step() with validation loss AFTER validation phase
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} (Depth: {train_loss_components['depth_loss']:.4f}, "
              f"Smooth: {train_loss_components['smoothness_loss']:.4f}, "
              f"Physics: {train_loss_components['physics_loss']:.4f})")
        
        print(f"Val Loss: {val_loss:.4f} (Depth: {val_loss_components['depth_loss']:.4f}, "
              f"Smooth: {val_loss_components['smoothness_loss']:.4f}, "
              f"Physics: {val_loss_components['physics_loss']:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_model(model, val_loader, criterion, device='cuda'):
    """
    Evaluate model performance on validation set
    
    Args:
        model: Trained model
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    depth_errors = {
        'abs_rel': 0.0,  # Absolute relative error
        'sq_rel': 0.0,   # Square relative error
        'rmse': 0.0,     # Root mean square error
        'rmse_log': 0.0, # Root mean square error in log space
        'a1': 0.0,       # Threshold accuracy (δ < 1.25)
        'a2': 0.0,       # Threshold accuracy (δ < 1.25²)
        'a3': 0.0        # Threshold accuracy (δ < 1.25³)
    }
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            # Move data to device
            images = sample['image'].to(device)
            target_depth = sample['depth'].to(device)
            
            # Forward pass
            outputs = model(images)
            pred_depth = outputs['depth']
            
            # Compute loss
            targets = {'depth': target_depth}
            loss, _ = criterion(outputs, targets, images)
            total_loss += loss.item()
            
            # Compute depth metrics
            pred = pred_depth.cpu().numpy()
            gt = target_depth.cpu().numpy()
            
            # Mask out invalid depths (zeros in ground truth)
            mask = gt > 0
            
            # Compute metrics
            for i in range(pred.shape[0]):
                pred_i = pred[i][mask[i]]
                gt_i = gt[i][mask[i]]
                
                if len(pred_i) == 0:
                    continue
                
                # Scale-invariant alignment (optional)
                ratio = np.median(gt_i) / np.median(pred_i)
                pred_i *= ratio
                
                # Compute errors
                thresh = np.maximum((gt_i / pred_i), (pred_i / gt_i))
                
                depth_errors['abs_rel'] += np.mean(np.abs(pred_i - gt_i) / gt_i)
                depth_errors['sq_rel'] += np.mean(((pred_i - gt_i) ** 2) / gt_i)
                depth_errors['rmse'] += np.sqrt(np.mean((pred_i - gt_i) ** 2))
                depth_errors['rmse_log'] += np.sqrt(np.mean((np.log(pred_i) - np.log(gt_i)) ** 2))
                
                # Threshold accuracies
                depth_errors['a1'] += np.mean((thresh < 1.25).astype(np.float32))
                depth_errors['a2'] += np.mean((thresh < 1.25 ** 2).astype(np.float32))
                depth_errors['a3'] += np.mean((thresh < 1.25 ** 3).astype(np.float32))
    
    # Average metrics
    total_loss /= len(val_loader)
    for k in depth_errors:
        depth_errors[k] /= len(val_loader.dataset)
    
    # Create full metrics dictionary
    metrics = {
        'loss': total_loss,
        **depth_errors
    }
    
    return metrics

def visualize_predictions(model, val_loader, num_samples=5, device='cuda', save_dir='visualizations'):
    """
    Visualize depth predictions on validation set
    
    Args:
        model: Trained model
        val_loader: DataLoader for validation set
        num_samples: Number of samples to visualize
        device: Device to run inference on
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if i >= num_samples:
                break
            
            # Move data to device
            images = sample['image'].to(device)
            target_depth = sample['depth']
            
            # Forward pass
            outputs = model(images)
            pred_depth = outputs['depth']
            
            # Get sparse points (if available)
            sparse_points = outputs.get('sparse_points', None)
            
            # Move tensors to CPU for visualization
            images_np = images.cpu().numpy()
            target_depth_np = target_depth.cpu().numpy()
            pred_depth_np = pred_depth.cpu().numpy()
            
            # Create visualization for each sample in batch
            for b in range(images_np.shape[0]):
                # Convert to correct format for visualization
                image = np.transpose(images_np[b], (1, 2, 0))  # [H, W, 3]
                gt_depth = target_depth_np[b, 0]               # [H, W]
                pred_depth = pred_depth_np[b, 0]               # [H, W]
                
                # Apply colormap to depth maps
                gt_depth_colored = visualize_depth(gt_depth)
                pred_depth_colored = visualize_depth(pred_depth)
                
                # Create figure with subplots
                plt.figure(figsize=(15, 5))
                
                # RGB image
                plt.subplot(1, 3, 1)
                plt.title("RGB Input")
                plt.imshow(image)
                plt.axis('off')
                
                # Ground truth depth
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Depth")
                plt.imshow(gt_depth_colored)
                plt.axis('off')
                
                # Predicted depth
                plt.subplot(1, 3, 3)
                plt.title("Predicted Depth")
                plt.imshow(pred_depth_colored)
                plt.axis('off')
                
                # If sparse points are available, visualize them
                if sparse_points is not None:
                    points = sparse_points[b].cpu().numpy()
                    h, w = pred_depth.shape
                    for p in points:
                        x, y = int(p[0] * w), int(p[1] * h)
                        if 0 <= x < w and 0 <= y < h:
                            plt.subplot(1, 3, 3)
                            plt.plot(x, y, 'r.', markersize=1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{i}_{b}.png"))
                plt.close()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DepthAugmentation:
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            depth = torch.flip(depth, [2])
        
        # Random brightness/contrast/gamma for RGB
        if random.random() > 0.5:
            brightness = 0.8 + random.random() * 0.4  # 0.8-1.2
            image = image * brightness
            image = torch.clamp(image, 0, 1)
            
        # Add color jitter
        if random.random() > 0.5:
            hue_factor = random.uniform(-0.1, 0.1)
            image = transforms.functional.adjust_hue(image, hue_factor)
        
        # Random crops
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(int(image.shape[1]*0.8), int(image.shape[2]*0.8)))
            image = transforms.functional.crop(image, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            # Resize back to original size
            image = transforms.functional.resize(image, (192, 256))
            depth = transforms.functional.resize(depth, (192, 256))
            
        return {'image': image, 'depth': depth}
    
def analyze_error_distribution(model, val_loader, device):
    """Analyze where errors occur most frequently"""
    model.eval()
    error_by_depth = []
    
    with torch.no_grad():
        for sample in val_loader:
            images = sample['image'].to(device)
            target = sample['depth'].to(device)
            
            outputs = model(images)
            pred = outputs['depth']
            
            # Compute relative error
            valid_mask = (target > 0.1).float()
            rel_error = torch.abs(pred - target) / (target + 1e-6) * valid_mask
            
            # Record depth and error
            for b in range(images.shape[0]):
                depths = target[b].flatten().cpu().numpy()
                errors = rel_error[b].flatten().cpu().numpy()
                
                for d, e in zip(depths, errors):
                    if d > 0.1:  # Valid depth
                        error_by_depth.append((d, e))
    
    depths, errors = zip(*error_by_depth)
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, errors, alpha=0.1)
    plt.xlabel('Depth (m)')
    plt.ylabel('Relative Error')
    plt.title('Error Distribution by Depth')
    plt.savefig('error_analysis.png')

def save_complete_model(model, path):
    torch.save(model, path)
    print(f"Complete model saved to {path}")

def main():
    # Define parameters
    img_height = 192
    img_width = 256
    batch_size = 4
    num_epochs = 30
    learning_rate = 1e-4
    
    # Define paths
    base_dir = "bathroom/bathroom"
    checkpoints_dir = "checkpoints"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = PhysicsGuidedDepthEstimation(img_height=img_height, img_width=img_width)

    model.apply(weights_init)
    
    # Define transformations (none needed as we handle it in the dataset class)
    transforms = None
    
    # Create datasets
    train_dataset = NYUv2BookstoreDataset(
        base_dir=base_dir,
        split='train',
        transform=transforms,
        height=img_height,
        width=img_width
    )
    
    val_dataset = NYUv2BookstoreDataset(
        base_dir=base_dir,
        split='val',
        transform=DepthAugmentation(),
        height=img_height,
        width=img_width
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Define loss function and optimizer
    criterion = PhysicsGuidedLoss(alpha=1.0, beta=0.1, gamma=0.1)  # Prioritize depth loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Slightly higher learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Check if we should resume training
    start_epoch = 0
    resume_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pth')
    
    if os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train model
    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=checkpoints_dir
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Visualize some predictions
    print("Generating visualizations...")
    visualize_predictions(
        model=model,
        val_loader=val_loader,
        num_samples=5,
        device=device,
        save_dir="visualizations"
    )
    
    # Generate depth maps for test images
    print("Generating depth maps for test images...")
    generate_depth_maps(
        model=model,
        test_images_dir=base_dir,
        output_dir="depth_maps_output",
        img_height=img_height,
        img_width=img_width
    )
    
    print("Training and evaluation complete!")

    analyze_error_distribution(model, val_loader, device='cuda')

    complete_model_path = "depth_model_complete.pth"
    save_complete_model(model, complete_model_path)

if __name__ == "__main__":
    main()