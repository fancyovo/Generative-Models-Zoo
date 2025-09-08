import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
class ImageDataset(Dataset):
    def __init__(self, image_dir, target_size):
        """
        Custom dataset for loading PNG images
        
        Args:
            image_dir (str): Directory containing PNG images
            target_size (tuple): Target image size (height, width)
        """
        self.image_dir = image_dir
        self.target_size = target_size
        
        # Get all PNG files in the directory
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith('.png')]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG files found in directory: {image_dir}")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0,1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        
        # Apply transforms
        image = self.transform(image)
        
        return image
def create_image_dataloader(dir_path, target_size, batch_size):
    """
    Create a DataLoader for PNG images in a directory
    
    Args:
        dir_path (str): Path to directory containing PNG images
        target_size (tuple): Target image size (height, width), e.g., (256, 256)
        batch_size (int): Batch size for DataLoader
    
    Returns:
        torch.utils.data.DataLoader: DataLoader containing resized images
    """
    # Validate inputs
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("target_size must be a tuple of two integers")
    
    if target_size[0] != target_size[1]:
        raise ValueError("Both dimensions in target_size must be equal")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    # Create dataset
    dataset = ImageDataset(dir_path, target_size)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for better training
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster GPU transfer if using CUDA
    )
    
    return dataloader


if __name__ == '__main__':
    dataset = create_image_dataloader('../../../../../out2', (256, 256), 32)
    print(f"Dataset length: {len(dataset)}")
    for batch in dataset:
        print(f"Batch shape: {batch.shape}")  # Should be [batch_size, 3, 256, 256]

