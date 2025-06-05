import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
from multi_node_aux import plot_loss_curves, create_scheduler, get_gpu_memory, l1_regularization, SimpleNN
import copy

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model and loss function
# You can now control the number of additional layers!
num_layers = 1 # 1 * 5  # Change this to create more or fewer layers
hidden_size = 128 # 128*64  # Starting size for additional layers
num_shareds = 20
num_epochs = 20
weight_decay = 1e-3 # 1e-4
l1_lambda = 0

print(f"Creating model with {num_layers} additional layers...")
model = SimpleNN(num_additional_layers=num_layers, hidden_size=hidden_size).to(device)
criterion = nn.CrossEntropyLoss()

# Print total number of parameters
all_params = list(model.parameters())
total_param_count = sum(p.numel() for p in all_params)
print(f"\n\nTotal parameters in model: {total_param_count:,}")
print(f"Total parameter tensors: {len(all_params)}")
print(f"Model will be trained in {num_shareds} shareds of ~{int(100/num_shareds)}% each\n")

# Define a simple neural network


# Function to unfreeze specific 20% shared and freeze the rest
def unfreeze_shared(model, shared_idx, is_print, num_shareds, is_freeze=True):
    """
    Unfreeze a specific 20% shared of parameters and freeze the rest.
    shared_idx: 0-4 (for 5 shareds of 20% each)
    
    This function creates balanced shareds based on parameter count, not tensor count.
    """
    all_params = list(model.parameters())
    
    # Create parameter shareds based on cumulative parameter counts
    param_info = []
    cumulative_count = 0
    
    for i, param in enumerate(all_params):
        param_count = param.numel()
        param_info.append({
            'index': i,
            'param': param,
            'count': param_count,
            'cumulative_start': cumulative_count,
            'cumulative_end': cumulative_count + param_count
        })
        cumulative_count += param_count

    # Calculate total parameter count and target shared size
    total_param_count = cumulative_count # sum(p.numel() for p in all_params)
    target_shared_size = total_param_count // num_shareds  # 20% of total parameters

    
    # Determine which parameters belong to current shared
    shared_start = shared_idx * target_shared_size
    shared_end = (shared_idx + 1) * target_shared_size
    if shared_idx == num_shareds - 1:  # Last shared gets remaining parameters
        shared_end = total_param_count
    
    if is_freeze:
        # Freeze all parameters first
        for param in all_params:
            param.requires_grad = False
    
    # Unfreeze parameters that overlap with current shared
    shared_param_count = 0
    info_list = []
    for info in param_info:
        # Check if this parameter tensor overlaps with current shared
        if (info['cumulative_start'] < shared_end and info['cumulative_end'] > shared_start):
            if is_freeze:
                info['param'].requires_grad = True
            shared_param_count += info['count']
            # Compute statistics for this parameter
            param = info['param']
            param_data = param.data            
            # Store layer-specific stats
            info['stats'] = {
                #'shape': tuple(param_data.shape),
                'mean': param_data.mean().item(),
                'variance':  param_data.var().item(),
                'std': param_data.std().item(),
                'min': param_data.min().item(),
                'max': param_data.max().item(),
                'num_params': param_data.numel()
            }
            info_list.append(info)
    mean_val = 0
    variance = 0
    for info in info_list:
        mean_val += info['stats']['mean'] * info['stats']['num_params'] / shared_param_count
        variance += info['stats']['variance']
    
    # Count final trainable and frozen parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # Debug info for first few calls
    if shared_idx <= num_shareds - 1 and is_print:
        print(f"      Shared {shared_idx + 1} range: params {shared_start:,} to {shared_end:,}")
        print(f"      Target shared size: {target_shared_size:,}")
        print(f"      Approximate total_param_count={total_param_count} cumulative_count={cumulative_count} target_shared_size={target_shared_size}\n")
    
    freeze_model = None

    return freeze_model, num_trainable_params, num_frozen_params, mean_val, variance, shared_param_count



# Test function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            if l1_lambda > 0:
                test_loss += l1_regularization(model=model, l1_lambda=l1_lambda)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Combined training function with all three loops
def train_model_with_shareds(model, train_loader, test_loader, criterion, device, num_epochs=5, lr=0.001):
    """
    Complete training function with three nested loops:
    1. Epoch loop (outer)
    2. Batch loop (middle)  
    3. Shared loop (inner)
    """
    
    print("Starting training with shareded parameter updates...")
    print("=" * 60)

    optimizer_list = []
    freeze_model_list = []

    model.train()

    train_loader_iter = iter(train_loader)
    (data, target) = next(train_loader_iter)
    data, target = data.to(device), target.to(device)

    loss_map = {}
    for shared_idx in range(num_shareds):
            loss_map[shared_idx] = []
            # === SHARED SETUP ===
            # Unfreeze current 20% shared, freeze the other 80%
            unfreeze_shared(model, shared_idx, num_shareds=num_shareds,is_print=True, is_freeze=True)                
            # Create optimizer for only the trainable parameters in current shared
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)            
            scheduler = create_scheduler(optimizer, num_epochs=num_epochs)
            optimizer.zero_grad()  # Clear gradients from previous iterations            
            optimizer_list.append(optimizer)

    all_params = list(model.parameters())
    for param in all_params:
        param.requires_grad = False


    learning_rates = []

    # EPOCH LOOP (OUTER): Train for specified number of epochs
    for epoch in range(num_epochs):
        current_lr = optimizer_list[0].param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"\nEPOCH {epoch + 1}/{num_epochs} current_lr={current_lr}")
        print("=" * 40)
        
        model.train()  # Set model to training mode
        epoch_loss = 0.0  # Track total loss for this epoch
        epoch_correct = 0  # Track correct predictions for this epoch
        epoch_total = 0  # Track total samples for this epoch
        
        start_time = time.time()  # Track epoch timing
        
        # BATCH LOOP (MIDDLE): Process each batch in the dataset
        for batch_idx, (data, target) in enumerate(train_loader):
            # Select GPU 
            shared_idx = (int(batch_idx)) % num_shareds

            # Move data to device (GPU/CPU)
            data, target = data.to(device), target.to(device)
            
            batch_loss = 0.0  # Track total loss for this batch across all shareds
            
            optimizer = optimizer_list[shared_idx]

            # === SHARED SETUP ===

            # Unfreeze current 20% shared, freeze the other 80%: gradient and optimizer size: 1/num_shareds of total
            _, num_trainable_params, num_frozen_params, mean_val, variance, shared_param_count = \
                unfreeze_shared(model, shared_idx, num_shareds=num_shareds, is_print=batch_idx == 0, is_freeze=True)
                                        
            # === FORWARD PASS ===
            optimizer.zero_grad()  # Clear gradients from previous iterations

            output = model(data)  # Forward pass through network

            loss = criterion(output, target)

            if l1_lambda > 0:
                loss += l1_regularization(model=model, l1_lambda=l1_lambda) # Calculate loss
                        
            # === BACKWARD PASS ===
            loss.backward()  # Compute gradients for trainable parameters only
            
            # === PARAMETER UPDATE ===
            optimizer.step()  # Update the trainable parameters
                        
            # Print detailed memory info every 100 batches
            loss_map[shared_idx].append(loss.item())
            if batch_idx % 100 == 0:
                if shared_idx > num_shareds - 3:
                    print(f"    Shared {shared_idx + 1}/{num_shareds}:")
                    print(f"      Trainable params: {num_trainable_params:,} shared_param_count={shared_param_count}")
                    print(f"  *** mean_val={mean_val:.5f}, variance={variance:.4f}, shared_param_count={shared_param_count}")
                    print(f"      Shared loss:       {loss.item():.6f}")
            
            # Accumulate batch loss
            batch_loss += loss.item()
            
            # === BATCH STATISTICS ===
            # Calculate average loss across all 5 shareds for this batch
            avg_batch_loss = batch_loss / num_shareds
            epoch_loss += avg_batch_loss
            
            # Calculate accuracy using final output (from last shared)
            pred = output.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_total += target.size(0)
            
            # Print batch summary every 200 batches
            if batch_idx % 200 == 0:
                running_acc = 100. * epoch_correct / epoch_total
                print(f"  Batch {batch_idx} Summary:")
                print(f"    Avg loss ({num_shareds} shareds): {avg_batch_loss:.6f}")
                print(f"    Running accuracy:    {running_acc:.2f}%")
        
        # === EPOCH STATISTICS ===
        # Calculate epoch averages
        epoch_avg_loss = epoch_loss / len(train_loader)
        epoch_accuracy = 100. * epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        
        # Test the model
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss:     {epoch_avg_loss:.4f}")
        print(f"  Train Accuracy: {epoch_accuracy:.2f}%")
        print(f"  Test Loss:      {test_loss:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.2f}%")
        print(f"  Time:           {epoch_time:.2f}s")
        print(f"  Final GPU Mem:  {get_gpu_memory():.2f} MB")

        loss_lists = [loss_map[shared_idx] for shared_idx in range(num_shareds)]
        labels = [f"Shared {i+1}" for i in range(num_shareds)]
        plot_loss_curves(loss_lists, labels=labels, 
                        title="Training Loss Curves", 
                        xlabel="Iteration", 
                        ylabel="Loss", 
                        figsize=(10, 6),
                        fname="shared_loss_curves")        

# Run the training
train_model_with_shareds(model, train_loader, test_loader, criterion, device, num_epochs=num_epochs, lr=0.001)

print("\nTraining completed!")
print(f"Final GPU memory usage: {get_gpu_memory():.2f} MB")