from PIL import Image
import fnmatch
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from diffusers.models import AutoencoderKL

import torchvision.utils as vutils

import matplotlib.pyplot as plt
import imageio.v3 as imageio
from torchvision.transforms import functional as F

def adjust_tensor_size(tensor, target_size=15):
    """
    Adjusts a tensor of shape [1, 1, x, 4096] to [1, 1, target_size, 4096]
    by padding when x < target_size and truncating when x > target_size.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [1, 1, x, 4096]
        target_size (int): Target size for the third dimension (default: 15)
    
    Returns:
        torch.Tensor: Adjusted tensor of shape [1, 1, target_size, 4096]
    """
    # Get the current size of the third dimension
    if len(tensor.shape) == 4:
        current_size = tensor.shape[2]
        
        # If current size is already the target size, return as is
        if current_size == target_size:
            return tensor
        elif current_size < target_size:
            result = torch.zeros(tensor.shape[0], tensor.shape[1], target_size, tensor.shape[3], dtype=tensor.dtype, device=tensor.device)
            # Copy the existing data
            result[:, :, :current_size, :] = tensor
            return result
        return tensor[:, :, :target_size, :]
    elif len(tensor.shape) == 3:
        current_size = tensor.shape[1]
        if current_size == target_size:
            return tensor
        elif current_size < target_size:
            result = torch.zeros(tensor.shape[0], target_size, tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
            #  Copy the existing data
            result[:, :current_size, :] = tensor
            return result
        return tensor[:, :target_size, :]
    elif len(tensor.shape) == 2:
        current_size = tensor.shape[0]
        if current_size == target_size:
            return tensor
        elif current_size < target_size:
            result = torch.zeros(target_size, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
            result[:current_size,:] = tensor
            return result
        return tensor[:target_size,:]
    else:
        raise ValueError(f"Unknown tensor shape: {tensor.shape}")
        
class LatentDataset(Dataset):
    def __init__(self, latent_dir, text_embed_dir, latent_pattern = "*.jpg", text_embed_pattern = "*_llava_emb_44.npy", transform=None, return_tuple=['file_name','latent', 'text_embed','edge_embed'], embed_text_target_size=15):
        
        self.latent_pattern = latent_pattern
        self.text_embed_pattern = text_embed_pattern

        self.latent_dir = latent_dir
        self.text_embed_dir = text_embed_dir
        
        self.latent_files = sorted(os.listdir(latent_dir))
        self.latent_files = [file for file in self.latent_files if fnmatch.fnmatch(file, latent_pattern)]
            
        self.text_embed_files = None
        self.text_embed_pattern = text_embed_pattern
        if text_embed_dir is not None:
            self.text_embed_files = sorted(os.listdir(text_embed_dir))
            self.text_embed_files = [file for file in self.text_embed_files if fnmatch.fnmatch(file, text_embed_pattern)]
            if not (len(self.latent_files) == len(self.text_embed_files)):
                print (f"Latent and text embed files must have the same length: {len(self.latent_files)} != {len(self.text_embed_files)}")

                # remove missing files
                set_embed_files = set(self.text_embed_files)
                removed_files = [file for file in self.latent_files if f"{file[:1-len(self.latent_pattern)]}{self.text_embed_pattern[1:]}" not in set_embed_files]
                self.latent_files = [file for file in self.latent_files if f"{file[:1-len(self.latent_pattern)]}{self.text_embed_pattern[1:]}" in set_embed_files]
                
                
                set_latent_files = set(self.latent_files)
                self.text_embed_files = [file for file in self.text_embed_files if f"{file[:1-len(self.text_embed_pattern)]}{self.latent_pattern[1:]}" in set_latent_files]
                assert len(self.latent_files) == len(self.text_embed_files)
                print (f"Removed files from text embed and latent files:{removed_files} ")

        self.transform = transform
        self.return_tuple = return_tuple
        self.embed_text_target_size = embed_text_target_size
    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file_name = self.latent_files[idx]
        fname = os.path.join(self.latent_dir, latent_file_name)
        
        if self.latent_pattern == "*.npy":
            # embedding
            latent = np.load(fname)
        elif self.latent_pattern == "*.jpg":
            # image
            latent = imageio.imread(fname) # ndarray
            # Convert numpy array to PIL Image
            latent = Image.fromarray(np.uint8(latent))
        elif self.latent_pattern == "*_embed_right.npy":
            latent = np.load(fname)      
        else:
            raise ValueError(f"Unknown latent pattern: {self.latent_pattern}")
        
        if self.text_embed_files is None:
            # image uses transform
            if self.transform is not None:
                latent = self.transform(latent) # torch.tensor(latent))
            if 'file_name' in self.return_tuple:
                return latent, latent_file_name
            else:
                return latent
        
        # text embed and image latent embed
        
        assert self.latent_pattern == "*_embed_right.npy"
        # 10004_AB_embed_right.npy -> 10004_AB_44_.npy
        # remobe '*' with 1: 
        text_embed_file = f"{latent_file_name[:1-len(self.latent_pattern)]}{self.text_embed_pattern[1:]}" 
        # add path
        file_name = os.path.join(self.text_embed_dir, text_embed_file)
        # self.text_embed_files[idx]
        text_embed = np.load(file_name) # 1,11,4096
        text_embed = text_embed.squeeze(0) # 11,4096
        # adjust text emb4ed to same shapes
        text_embed = torch.tensor(text_embed)
        text_embed =adjust_tensor_size(text_embed, target_size=self.embed_text_target_size)

        # edge embed
        edge_embed_file = f"{latent_file_name[:1-len(self.latent_pattern)]}_embed_left.npy"
        file_name = os.path.join(self.text_embed_dir, edge_embed_file)
        edge_embed = np.load(file_name)
        
        assert self.transform is None
        assert 'text_embed' in self.return_tuple, "text_embed must be in return_tuple"
        assert 'latent' in self.return_tuple, "latent must be in return_tuple"

        if 'file_name' not in self.return_tuple: # "file_name  not be in return_tuple"
            return torch.tensor(latent), text_embed, torch.tensor(edge_embed)
        
        return torch.tensor(latent), text_embed, torch.tensor(edge_embed), latent_file_name
    
def create_latent_dataset(latent_dir, text_embed_dir, latent_pattern = "*.npy", text_embed_pattern = "*_44.npy"):
    return LatentDataset(latent_dir, text_embed_dir, latent_pattern, text_embed_pattern)


class CropRightHalf(object):
    def __call__(self, img):
        # Crop right half (coordinates are left, top, right, bottom)
        # For 512x256 image, right half is (256, 0, 512, 256)
        return F.crop(img, top=0, left=256, height=256, width=256)

class CropLeftHalf(object):
    def __call__(self, img):
        # Crop right half (coordinates are left, top, right, bottom)
        # For 512x256 image, right half is (256, 0, 512, 256)
        return F.crop(img, top=0, left=0, height=256, width=256)

# Create a transform pipeline
transform_example = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    CropRightHalf(),        # Crop right half
    # Add any additional transforms here
])



def run_latent_generation(dataset_dir, output_latent_dir, transform, input_pattern="*.jpg", output_suffix="_embed_right.npy"):

    # image_files = sorted(os.listdir(dataset_dir))
    # image_files = [file for file in image_files if fnmatch.fnmatch(file, input_pattern)]

    #dataset = ImageFolder(dataset_dir, transform=transform)
    dataset = LatentDataset(latent_dir=dataset_dir, text_embed_dir=None, transform=transform, latent_pattern=input_pattern, text_embed_pattern="*_44.npy", return_tuple=['file_name','latent'])
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    len_pattern = len(input_pattern) - 1
    img_index = 0
    with torch.no_grad():
        for x, file_name in tqdm(data_loader, leave=False):
            
            x = x.to(device)
            
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda:0'):
                # Map input images to latent space + normalize latents:
                latent_features = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent_features = latent_features.detach().cpu()  # (bs, 4, image_size//8, image_size//8) ([8, 4, 32, 32])
            
            for ind, latent in enumerate(latent_features.split(1, 0)):
                # torch.Size([1, 4, 32, 32])
                image_file = file_name[ind]
                fn = dataset_dir + f'/{image_file[:-len_pattern]}{output_suffix}'
                np.save(fn, latent.squeeze(0).numpy())
                img_index += 1

    return img_index

def run(data_loader, latent_save_dir, suffix="_right"):
    img_index = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                # Map input images to latent space + normalize latents:
                latent_features = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent_features = latent_features.detach().cpu()  # (bs, 4, image_size//8, image_size//8)
    
            for latent in latent_features.split(1, 0):
                fn = latent_save_dir + f'/{img_index}_{suffix}.npy'
                np.save(fn, latent.squeeze(0).numpy())
                img_index += 1

def create_transform(image_size):
    transform_right = transforms.Compose([  
        transforms.Resize(image_size),
        CropRightHalf(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    transform_left = transforms.Compose([
        transforms.Resize(image_size),
        CropLeftHalf(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform, transform_right, transform_left

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    gpu_indx  = 0
    device = torch.device(gpu_indx if use_cuda else "cpu")
    
    batch_size = 10 # 8 # 32

    image_size = 256
    
    transform, transform_right, transform_left = create_transform(image_size)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)


    EDGEBAGS = True

    # Setup data:
    if EDGEBAGS:
        nm = "edges2shoes" # edges2handbags
        if False:
            dataset_dir = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets/{nm}/" # /class_A" #"."
            latent_save_dir = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets/{nm}_latent" # "."
            latent_save_dir_right = latent_save_dir + "_right"
            latent_save_dir_left = latent_save_dir + "_left"

    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    

    if EDGEBAGS:
        # 512 x 256
        nm = "edges2shoes" # edges2handbags
        #dataset_dir_train = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets/{nm}/train" # /class_A" #"."
        dataset_dir_train = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets/{nm}/test" # /class_A" #"."
        dataset_dir_test = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets/{nm}/test_new_color" # /class_A" #"."
        if False: # save separate directories for right and left images
            latent_save_dir_right = f"{dataset_dir}_latent_right"
            latent_save_dir_left = f"{dataset_dir}_latent_left"
            os.makedirs(latent_save_dir_right, exist_ok=True)
            os.makedirs(latent_save_dir_left, exist_ok=True)
            dataset_right = ImageFolder(dataset_dir, transform=transform_right) # must havve class_A and class_B folders
            dataset_left = ImageFolder(dataset_dir, transform=transform_left)
    else:    
        dataset_dir = "/home/roman/PycharmProjects/jobs/dandy/pytorch_tutorials/datasets/CELEBA_HQ_256" 
        latent_save_dir = "/home/roman/PycharmProjects/jobs/dandy/pytorch_tutorials/datasets/CELEBA_HQ_256_LATENT" 
        dataset = ImageFolder(dataset_dir, transform=transform)
        
    #if False:        
    #    data_set_root = "/media/luke/Quick_Storage/Data/CelebAHQ/image_latents"        
    #    trainset = LatentDataset(data_set_root)
    #else:    
    #    dir_latent = latent_save_dir_right
    #     trainset = LatentDataset(dir_latent)# data_set_root)
        

    # Create latents for right and left images: save into SAME directory as _right.npy and _left.npy

    if True:
        if True: # debug
            num_workers = 1
            pin_memory = False

            dataset_right = LatentDataset(latent_dir=dataset_dir_test, text_embed_dir=None, latent_pattern="*.jpg", transform=transform_right, return_tuple=['file_name','latent'])
            dataset_left = LatentDataset(latent_dir=dataset_dir_test, text_embed_dir=None, latent_pattern="*.jpg",  transform=transform_left, return_tuple=['file_name','latent'])
            print(f"{len(dataset_right)},{len(dataset_left)}")

            data_loader_left = DataLoader(dataset_left, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            data_loader_right = DataLoader(dataset_right, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

            #dataiter_left = iter(data_loader_left)
            #dataiter_right = iter(data_loader_right)
            
            #for ind, dataiter in enumerate([dataiter_left, dataset_right]):
            for ind, dataloader in enumerate([data_loader_left, data_loader_right]):
                dataiter = iter(dataloader)
                images, file_names = next(dataiter) # images, labels = next(dataiter_left)

                print(f"image = {len(images)} {images.shape}")# labels= {labels.shape} {labels[:10]}")

                fake_sample = images
                plt.figure(figsize = (20, 10))
                out = vutils.make_grid(fake_sample.detach().float().cpu(), nrow=8, normalize=True)
                _ = plt.imshow(out.numpy().transpose((1, 2, 0)))
                plt.axis('off')  # Hide the axes for cleaner visualization
                plt.tight_layout()  # Adjust the padding
                # plt.show()
                plt.savefig(f'output_image_{ind}.png')  # Save to a file instead of showing
                plt.close()  # Close the figure to free memory        

            x = images.to(device)
            #with torch.cuda.amp.autocast():
            with torch.amp.autocast("cuda:0"):    
                # Map input images to latent space + normalize latents:
                latent_features = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent_features = latent_features.detach().cpu()  # (bs, 4, image_size//8, image_size//8)

        for dataset_dir in [dataset_dir_test, dataset_dir_train]:
            print(f"dataset_dir={dataset_dir.__len__}")
            if True:
                num_right =run_latent_generation(dataset_dir=dataset_dir, output_latent_dir=dataset_dir, transform=transform_right, input_pattern="*.jpg", output_suffix="_embed_right.npy")
            else:
                run(data_loader_right, latent_save_dir_right, suffix="_right")

            if True:
                num_left =run_latent_generation(dataset_dir=dataset_dir, output_latent_dir=dataset_dir, transform=transform_left,  input_pattern="*.jpg", output_suffix="_embed_left.npy")
            else:
                run(data_loader_left, latent_save_dir_left, suffix="_left")

        print(f"run_latent_generation: num_right={num_right} num_left={num_left} dataset_right={len(dataset_right)} dataset_left={len(dataset_left)}")
        quit()

    # 
    # Dataset: return latent npy, text_embed npy, ledt_image (edges) npy.
    #

    # Example usage
    # Create a sample tensor
    x = 10  # Example value
    sample_tensor = torch.randn(3, 7, x, 4096)

    # Adjust to target size
    adjusted_tensor = adjust_tensor_size(sample_tensor)
    print(f"Original shape: {sample_tensor.shape} Adjusted shape: {adjusted_tensor.shape}")

    # Try with x > 15
    sample_tensor_large = torch.randn(3, 20, 4096)
    adjusted_tensor_large = adjust_tensor_size(sample_tensor_large)
    print(f"Original shape (large): {sample_tensor_large.shape} Adjusted shape (large): {adjusted_tensor_large.shape}")



    dataset = LatentDataset(latent_dir=dataset_dir, text_embed_dir=dataset_dir, latent_pattern="*_embed_right.npy", text_embed_pattern="*_llava_emb_44.npy", transform=None, return_tuple=['text_embed','latent', 'edge_embed'])
    print(len(dataset))

    # debug
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    data_loader_iter = iter(data_loader)
    latent, text_embed, edge_embed = next(data_loader_iter)
    print(f"latent.shape={latent.shape} text_embed.shape={text_embed.shape} edge_embed.shape={edge_embed.shape}")
    
    