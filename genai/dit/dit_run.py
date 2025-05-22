import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import copy
import math
from tqdm.notebook import trange, tqdm
from PIL import Image

from diffusers.models import AutoencoderKL

from dit_dataset import LatentDataset
# training parameters
batch_size =  16 # 32
lr = 2e-5

train_epoch = 1200

# data_loader
latent_size = 32

data_set_root = "../../datasets"

use_cuda = torch.cuda.is_available()
gpu_indx  = 0
device = torch.device(gpu_indx if use_cuda else "cpu")




def extract_patches(image_tensor, patch_size=8):
    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()
    
    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    
    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)
    
    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    
    return unfolded


def reconstruct_image(patch_sequence, image_shape, patch_size=8):
    """
    Reconstructs the original image tensor from a sequence of patches.

    Args:
        patch_sequence (torch.Tensor): Sequence of patches with shape
        BS x L x (C x patch_size x patch_size)
        image_shape (tuple): Shape of the original image tensor (bs, c, h, w).
        patch_size (int): Size of the patches used in extraction.

    Returns:
        torch.Tensor: Reconstructed image tensor.
    """
    bs, c, h, w = image_shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    # Reshape the patch sequence to match the unfolded tensor shape
    unfolded_shape = (bs, num_patches_h, num_patches_w, patch_size, patch_size, c)
    patch_sequence = patch_sequence.view(*unfolded_shape)
    
    # Transpose dimensions to match the original image tensor shape
    patch_sequence = patch_sequence.permute(0, 5, 1, 3, 2, 4).contiguous()
    
    # Reshape the sequence of patches back into the original image tensor shape
    reconstructed = patch_sequence.view(bs, c, h, w)
    
    return reconstructed

class ConditionalNorm2d(nn.Module):
    def __init__(self, hidden_size, num_features):
        super(ConditionalNorm2d, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.fcw = nn.Linear(num_features, hidden_size)
        self.fcb = nn.Linear(num_features, hidden_size)

    def forward(self, x, features):
        bs, s, l = x.shape # 8,256,768
        
        out = self.norm(x) # 8,256,768
        w = self.fcw(features).reshape(bs, 1, -1) # 8,1,768
        b = self.fcb(features).reshape(bs, 1, -1) # 8,1,768

        return w * out + b # 8,256,768

    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
# Transformer block with self-attention
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, num_features=128):
        # Initialize the parent nn.Module
        super(TransformerBlock, self).__init__()
        
        # Layer normalization to normalize the input data
        self.norm = nn.LayerNorm(hidden_size)
        
        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                                    batch_first=True, dropout=0.0)
        
        # Multi-head cross-attention mechanism for text embeddings
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                               batch_first=True, dropout=0.0)
        # Layer normalization for cross-attention
        self.norm2_cross_attn = nn.LayerNorm(hidden_size)
                
        # Another layer normalization
        self.con_norm = ConditionalNorm2d(hidden_size, num_features)
        
        # Multi-layer perceptron (MLP) with a hidden layer and activation function
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
                
    def forward(self, x, features, text_embeddings=None):
        # Apply the first layer normalization
        norm_x = self.norm(x)
        
        # Apply multi-head attention and add the input (residual connection)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x 
        
        # Apply cross-attention with text embeddings if provided
        if text_embeddings is not None:
            norm_x = self.norm2_cross_attn(x)
            x = self.cross_attn(norm_x, text_embeddings, text_embeddings)[0] + x

        # Apply the second layer normalization
        norm_x = self.con_norm(x, features) # x = 8,256,768 features = 8,128 
        
        # Pass through the MLP and add the input (residual connection)
        x = self.mlp(norm_x) + x
        
        return x

# Text Embedding Projector
class TextProjector(nn.Module):
    def __init__(self, text_embed_dim=4096, hidden_size=128):
        super(TextProjector, self).__init__()
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, text_embeddings):
        return self.text_projection(text_embeddings)
    
# Define a Vision Encoder module for the Diffusion Transformer
class DiT(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, 
                 hidden_size=128, num_features=128, 
                 num_layers=3, num_heads=4,
                 text_embed_dim=4096):
        super(DiT, self).__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU()
        )

        # Text embedding projector
        self.text_projector = TextProjector(text_embed_dim, hidden_size)        
        
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
        
        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_size, channels_in * patch_size * patch_size)
                
    def forward(self, image_in, index, text_embeddings=None):  
        # Get timestep embedding
        index_features = self.time_mlp(index) # [8, 128]

        # Process text embeddings if provided
        projected_text = None
        if text_embeddings is not None:
            # Assuming text_embeddings has shape [B, seq_len, text_embed_dim]
            # ex. 15 tokens, 4096 dim
            assert len(text_embeddings.shape) == 3
            B, seq_len, text_embed_dim_debug = text_embeddings.shape # == text_embed_dim
            # Project text embeddings to hidden size
            projected_text = self.text_projector(text_embeddings.reshape(B * seq_len, -1)) # 120, 128
            projected_text = projected_text.reshape(B, seq_len, -1)  # 8,15,128 [B, seq_len, hidden_size]

        # Split input into patches
        patch_seq = extract_patches(image_in, patch_size=self.patch_size) # ([8, 256, 16]
        patch_emb = self.fc_in(patch_seq) # ([8, 256, 768]

        # Add a unique embedding to each token embedding
        embs = patch_emb + self.pos_embedding #  ([8, 256, 768]
        
        # Pass the embeddings through each Transformer block
        for block in self.blocks:
            embs = block(embs, index_features, projected_text)
        
        # Project to output
        image_out = self.fc_out(embs) # ch.Size([8, 256, 16])
        
        # Reconstruct the input from patches and return result
        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size)

def cosine_alphas_bar(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar[:timesteps]

def noise_from_x0(curr_img, img_pred, alpha):
    return (curr_img - alpha.sqrt() * img_pred)/((1 - alpha).sqrt() + 1e-4)

def cold_diffuse(diffusion_model, sample_in, total_steps, text_embeddings=None, start_step=0):
    diffusion_model.eval()
    bs = sample_in.shape[0]
    alphas = torch.flip(cosine_alphas_bar(total_steps), (0,)).to(device)
    random_sample = copy.deepcopy(sample_in) # ([8, 4, 32, 32]
    with torch.no_grad():
        for i in trange(start_step, total_steps - 1):
            index = (i * torch.ones(bs, device=sample_in.device)).long() # [8]

            img_output = diffusion_model(random_sample, index, text_embeddings) #

            noise = noise_from_x0(random_sample, img_output, alphas[i])
            x0 = img_output

            rep1 = alphas[i].sqrt() * x0 + (1 - alphas[i]).sqrt() * noise
            rep2 = alphas[i + 1].sqrt() * x0 + (1 - alphas[i + 1]).sqrt() * noise

            random_sample += rep2 - rep1

        index = ((total_steps - 1) * torch.ones(bs, device=sample_in.device)).long()
        img_output = diffusion_model(random_sample, index, text_embeddings)

    return img_output


def show_latent_dit(dit, timesteps, latent_size, device, text_embeddings=None):
    #
    # non trained model!!!!
    #

    latent_noise = 0.95 * torch.randn(8, 4, latent_size, latent_size, device=device)

    with torch.no_grad():
        #with torch.cuda.amp.autocast():
        with torch.amp.autocast("cuda:0"): 
            # fake_latents = cold_diffuse(u_net, latent_noise, total_steps=timesteps)
            fake_latents = cold_diffuse(dit, latent_noise, total_steps=timesteps, text_embeddings=text_embeddings)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    with torch.no_grad():
        #with torch.cuda.amp.autocast():
        with torch.amp.autocast("cuda:0"): 
            fake_sample = vae.decode(fake_latents / 0.18215).sample
            
    plt.figure(figsize = (20, 10))

    out = vutils.make_grid(fake_sample[:8].detach().float().cpu(), nrow=4, normalize=True)

    _ = plt.imshow(out.numpy().transpose((1, 2, 0)))
    return fake_sample


def validate_dit(total_steps, latent_size):
    # Initialize model
    model = DiT(
        image_size=latent_size,            # Size of input images
        channels_in=4,            # Number of input channels
        patch_size=2,            # Size of image patches
        hidden_size=768,          # Model's hidden dimension
        # num_features=64,         # Timestep embedding dimension
        num_layers=3,             # Number of transformer blocks
        num_heads=4,              # Number of attention heads
        text_embed_dim=4096       # Text embedding dimension
    ).to(device)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    
    starting_noise = 0.95 * torch.randn(8, 4, latent_size, latent_size, device=device)    

    # Example inputs
    images = torch.randn(8, 4, 32, 32).to(device)         # [B, C, H, W]
    timesteps = torch.randint(0, total_steps, (8,)).to(device)   # [B]
    text_embeds = torch.randn(8, 15, 4096).to(device)     # [B, seq_len, embed_dim]

    # Forward pass with text conditioning
    output = model(images, timesteps, text_embeds) # 8,4,32,32

    # For inference
    generated = cold_diffuse(model, starting_noise, total_steps=total_steps, text_embeddings=text_embeds)

    return generated

def fake(dit, text_embed_cpu, timesteps, latent_noise=None):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    if latent_noise is None:
        latent_noise = 0.95 * torch.randn(text_embed_cpu.shape[0], 4, latent_size, latent_size, device=device)
    with torch.no_grad():
        with torch.amp.autocast('cuda:0'):
        # with torch.cuda.amp.autocast():
            fake_latents = cold_diffuse(dit, latent_noise, total_steps=timesteps, text_embeddings=text_embed_cpu)
            fake_sample = vae.decode(fake_latents / 0.18215).sample
    return fake_sample, latent_noise
    
def load_model(fn="latent_dit_100.pt", device="cpu"):
    cp = torch.load(fn)
    
    # network
    dit_100 = DiT(latent_size, channels_in=latents.shape[1], patch_size=patch_size, 
                hidden_size=768, num_layers=10, num_heads=8).to(device)
    
    
    dit_100.load_state_dict(cp["model_state_dict"])
    
    optimizer_100 = optim.Adam(dit_100.parameters(), lr=lr)
    
    optimizer_100.load_state_dict(cp["optimizer_state_dict"])
    
    loss_log_100 = cp["train_data_logger"]
    
    start_epoch = cp["epoch"]
    
    print(f"start_epoch={start_epoch}")

    return dit_100, optimizer_100, loss_log_100, start_epoch

def save_image(start_epoch, fake_sample_100, fn):
    plt.figure(figsize = (20, 10))
    out = vutils.make_grid(fake_sample_100.detach().float().cpu(), nrow=4, normalize=True)
    _ = plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.axis('off')  # Hide the axes for cleaner visualization
    plt.tight_layout()  # Adjust the padding
    # plt.show()
    plt.savefig(fn)  # Save to a file instead of showing
    plt.close()  # Close the figure to free memory       
    print(f"start_epoch={start_epoch} fn={fn}")

def save_image_title(start_epoch, fake_sample_100, fn, titles=None):
    plt.figure(figsize=(20, 10))
    
    # Create the grid of images
    out = vutils.make_grid(fake_sample_100.detach().float().cpu(), nrow=4, normalize=True)
    
    # Display the image grid
    ax = plt.gca()
    ax.imshow(out.numpy().transpose((1, 2, 0)))
    ax.axis('off')  # Hide the axes for cleaner visualization
    
    # Add titles if provided
    if titles is not None:
        # Calculate grid dimensions
        batch_size = fake_sample_100.size(0)
        nrow = 4
        ncol = (batch_size + nrow - 1) // nrow  # Calculate number of rows needed
        
        # Get image dimensions from the grid
        grid_height, grid_width = out.shape[1], out.shape[2]
        img_height = grid_height // ncol
        img_width = grid_width // nrow
        
        # Add title for each image
        for i, title in enumerate(titles[:batch_size]):  # Limit to actual number of images
            row = i // nrow
            col = i % nrow
            
            # Calculate position for the title (below each image)
            x = col * img_width + img_width // 2
            y = (row + 1) * img_height - 10  # Slightly above the bottom of each image
            
            # Add text with background for better visibility
            ax.text(x, y, title, ha='center', va='bottom', fontsize=10, 
                   color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight')  # bbox_inches='tight' ensures titles aren't cut off
    plt.close()
    print(f"start_epoch={start_epoch} fn={fn}")

def create_latent_dataset(dataset_dir, shuffle = True, return_tuple=['latent', 'text_embed','edge_embed']): 
    trainset = LatentDataset(latent_dir=dataset_dir,      text_embed_dir=dataset_dir,      latent_pattern="*_embed_right.npy", text_embed_pattern="*_llava_emb_44.npy", transform=None, return_tuple=return_tuple)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dataiter = iter(train_loader)
    return trainset, train_loader, dataiter


if __name__ == "__main__":

    timesteps = 500 # total_steps = 500
    patch_size = 2


    # non trained model!!!!
    if False:
        _ =validate_dit(timesteps, latent_size)

    nm = "edges2shoes" # edges2handbags
    model_root = f"/home/roman/PycharmProjects/jobs/dandy/pytorch_tutorials/section09_generation/solutions"
    data_root = f"/home/roman/PycharmProjects/jobs/dandy/pytorch-CycleGAN-and-pix2pix/datasets"
    dataset_dir =          f"{data_root}/{nm}/train" 
    dataset_dir_test     = f"{data_root}/{nm}/test" 
    dataset_dir_test_red = f"{data_root}/{nm}/test_new_color" 

    trainset, train_loader, dataiter = create_latent_dataset(dataset_dir)
    testset, test_loader, dataiter_test = create_latent_dataset(dataset_dir_test, shuffle=False, return_tuple=['latent', 'text_embed','edge_embed','file_name'])
    testset_red, test_loader_red, dataiter_test_red = create_latent_dataset(dataset_dir_test_red, shuffle=False, return_tuple=['latent', 'text_embed','edge_embed','file_name'])
    
    print(len(trainset), len(testset), len(testset_red))

    # Sample from the itterable object
    latents, text_embed, edge_embed = next(dataiter)
    print(f"latents={latents.shape} latent_size={latent_size} text_embed={text_embed.shape} edge_embed={edge_embed.shape}")
# Create a dataloader itterable object
    # Sample from the itterable object
    
    #_ = next(dataiter_test)
    #_ = next(dataiter_test_red)
    latents_test, text_embed_test, edge_embed_test, file_name_test = next(dataiter_test)
    latents_test_red, text_embed_test_red, edge_embed_test_red, file_name_test_red = next(dataiter_test_red)
    
    # read texts from file_name_test_red
    def read_text_from_file(file_name_list, dataset_dir_test):
        file_name_list = [fn[: -len("_embed_right.npy")] + ".txt" for fn in file_name_list]
        text_list = []
        for fn in file_name_list:
            with open(f"{dataset_dir_test}/{fn}", 'r') as file:
                text = file.read()
                text = text.replace('\n', '')
                text_list.append(text)
        return text_list
    text_list_test = read_text_from_file(file_name_test, dataset_dir_test)
    text_list_test_red = read_text_from_file(file_name_test_red, dataset_dir_test_red)
    # Create a dataloader itterable object
    # network
    dit = DiT(latent_size, channels_in=latents.shape[1], patch_size=patch_size, 
                hidden_size=768, num_layers=10, num_heads=8).to(device)

    # Adam optimizer
    optimizer = optim.Adam(dit.parameters(), lr=lr)

    # Scaler for mixed precision training
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    alphas = torch.flip(cosine_alphas_bar(timesteps), (0,)).to(device)

    # Let's see how many Parameters our Model has!
    num_model_params = 0
    for param in dit.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))


    # fake_sample = show_latent_dit(dit, timesteps, latent_size, device)
    checkpoint_fn = f"{model_root}/text_embed_latent_dit_100.pt"
    try:
        dit, optimizer, loss_log, start_epoch = load_model(f"{checkpoint_fn}", device=device)
    except:
        start_epoch = 0

    print(f"norm diff {torch.norm(text_embed_test_red - text_embed_test)} norm text_embed_test_red={torch.norm(text_embed_test_red)} norm text_embed_test={torch.norm(text_embed_test)}")
    fake_sample_100, latent_noise = fake(dit, text_embed_test_red.to(device), timesteps, latent_noise = None)

    num_images = 8 # 12
    save_image_title(start_epoch, fake_sample_100[:num_images], f"{model_root}/output_image_test_red_{start_epoch}.png", titles=text_list_test_red[:num_images])

    fake_sample_100, _ = fake(dit, text_embed_test.to(device), timesteps, latent_noise)
    save_image_title(start_epoch, fake_sample_100[:num_images], f"{model_root}/output_image_test_{start_epoch}.png", titles=text_list_test[:num_images] )

    
    loss_log = []
    mean_loss = 0


    # pbar = trange(start_epoch, train_epoch, leave=False, desc="Epoch")    

    # pbar = trange(0, 2, leave=False, desc="Epoch")    

    #pbar = trange(0, 100, leave=False, desc="Epoch")    
    pbar = trange(start_epoch, start_epoch+1000, leave=False, desc="Epoch")    
    # train_loader = torch.utils.data.DataLoader(trainset[:360], batch_size=batch_size, shuffle=True, num_workers=4)

    dit.train()

    for epoch in pbar:
        pbar.set_postfix_str('Loss: %.4f' % (mean_loss/len(train_loader)))
        mean_loss = 0

        for num_iter, (latents, text_embed, edge_embed) in enumerate(tqdm(train_loader, leave=False)):
            latents = latents.to(device)
            text_embed = text_embed.to(device)
            edge_embed = edge_embed.to(device)

            #the size of the current minibatch
            bs = latents.shape[0]

            rand_index = torch.randint(timesteps, (bs, ), device=device)
            random_sample = torch.randn_like(latents)
            alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)
            
            noise_input = alpha_batch.sqrt() * latents +\
            (1 - alpha_batch).sqrt() * random_sample
            
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
                latent_pred = dit(noise_input, rand_index, text_embed)
                loss = F.l1_loss(latent_pred, latents)
            
            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #log the generator training loss
            loss_log.append(loss.item())
            mean_loss += loss.item()
            # Quick save of the model every epoch
            if num_iter > 0 and num_iter % 200 == 0:
                print(f"Saving epoch {epoch} iter {num_iter} loss={loss_log[-1]} model to {checkpoint_fn}")
                torch.save({'epoch': epoch + 1,
                            'train_data_logger': loss_log,
                            'model_state_dict': dit.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, checkpoint_fn)
        torch.save({'epoch': epoch + 1,
                    'train_data_logger': loss_log,
                    'model_state_dict': dit.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_fn)
        

    # Plot loss
    plt.figure(figsize = (20, 10))
    plt.plot(loss_log[:])
    plt.tight_layout()  # Adjust the padding
    # plt.show()
    plt.savefig(f'loss_log.png')  # Save to a file instead of showing
    plt.close()  # Close the figure to free memory          


    fake_sample_100, latent_noise = fake(dit, text_embed_test_red.to(device), timesteps, latent_noise = None)
    save_image(start_epoch, fake_sample_100, f"{model_root}/output_image_test_red_{start_epoch}.png")

    fake_sample_100, _ = fake(dit, text_embed_test.to(device), timesteps, latent_noise)
    save_image(start_epoch, fake_sample_100, f"{model_root}/output_image_test_{start_epoch}.png")  

