import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, num_additional_layers=30, hidden_size=32):
        """
        Initialize neural network with configurable number of layers.
        
        Args:
            num_additional_layers (int): Number of additional linear layers to create
            hidden_size (int): Starting size for additional layers
        """
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.num_additional_layers = num_additional_layers
        
        # Original layers
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, hidden_size)  # Connect to additional layers
        
        # DYNAMIC: Create specified number of additional layers
        self.additional_linear_layers = nn.ModuleList()
        self.additional_relu_layers = nn.ModuleList()
        
        # Calculate layer sizes that gradually decrease
        layer_sizes = self._calculate_layer_sizes(hidden_size, num_additional_layers)
        
        # Create the specified number of linear layers
        for i in range(num_additional_layers):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Add linear layer and corresponding ReLU
            self.additional_linear_layers.append(nn.Linear(input_size, output_size))
            self.additional_relu_layers.append(nn.ReLU())
            
            print(f"Created additional layer {i+1}: {input_size} -> {output_size}")
        
        # Final output layer (always outputs 10 for MNIST)
        final_input_size = layer_sizes[-1]
        self.output_layer = nn.Linear(final_input_size, 10)
        
        print(f"Created output layer: {final_input_size} -> 10")
        print(f"Total additional layers created: {num_additional_layers}")
    
    def _calculate_layer_sizes(self, start_size, num_layers):
        """
        Calculate layer sizes that gradually decrease from start_size to a minimum.
        
        Args:
            start_size (int): Starting layer size
            num_layers (int): Number of layers to create
            
        Returns:
            list: Layer sizes including input and output sizes
        """
        if num_layers == 0:
            return [start_size]
        
        # Minimum size for the last layer before output
        min_size = max(8, 10)  # At least 8, but ensure it's >= 10 for final connection
        
        # Create gradually decreasing sizes
        sizes = [start_size]
        
        if num_layers == 1:
            sizes.append(min_size)
        else:
            # Calculate step size for gradual decrease
            step = max(1, (start_size - min_size) // num_layers)
            
            for i in range(1, num_layers + 1):
                new_size = max(min_size, start_size - (step * i))
                sizes.append(new_size)
        
        return sizes
    
    def forward(self, x):
        # Original forward pass
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # Now outputs to first additional layer
        
        # Forward through dynamically created additional layers
        for i in range(self.num_additional_layers):
            linear_layer = self.additional_linear_layers[i]
            relu_layer = self.additional_relu_layers[i]
            x = relu_layer(linear_layer(x))
        
        # Final output layer
        x = self.output_layer(x)
        
        return x
    

def l1_regularization(model, l1_lambda):
    """
    Calculate L1 regularization penalty for all model parameters.
    
    Args:
        model: PyTorch model
        l1_lambda (float): L1 regularization strength
    
    Returns:
        torch.Tensor: L1 penalty term
    """
    l1_penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:  # Only penalize trainable parameters
            l1_penalty += torch.sum(torch.abs(param))
    return l1_lambda * l1_penalty


# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to get GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    return 0

def create_scheduler(optimizer, num_epochs, scheduler_type='exponential', scheduler_params=None):
    """
    Create a learning rate scheduler based on the specified type and parameters.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type (str): Type of scheduler
        scheduler_params (dict): Parameters for the scheduler
        num_epochs (int): Total number of epochs (used for some schedulers)
    
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == 'none':
        return None
    
    # Set default parameters if none provided
    if scheduler_params is None:
        scheduler_params = {}
    
    if scheduler_type == 'step':
        # Default: decay by 0.1 every 30 epochs
        step_size = scheduler_params.get('step_size', max(30, num_epochs // 3))
        gamma = scheduler_params.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        # Default: decay by 0.95 every epoch
        gamma = scheduler_params.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        # Default: cosine annealing over all epochs
        T_max = scheduler_params.get('T_max', num_epochs)
        eta_min = scheduler_params.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'plateau':
        # Default: reduce by 0.5 when loss plateaus for 10 epochs
        patience = scheduler_params.get('patience', 10)
        factor = scheduler_params.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, verbose=True)
    
    elif scheduler_type == 'multistep':
        # Default: decay at 60% and 80% of total epochs
        milestones = scheduler_params.get('milestones', [int(0.6 * num_epochs), int(0.8 * num_epochs)])
        gamma = scheduler_params.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'linear':
        # Default: linear decay from 1.0 to 0.1 over all epochs
        start_factor = scheduler_params.get('start_factor', 1.0)
        end_factor = scheduler_params.get('end_factor', 0.1)
        total_iters = scheduler_params.get('total_iters', num_epochs)
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, 
                                         end_factor=end_factor, total_iters=total_iters)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
def plot_loss_curves(loss_lists, labels=None, title="Training Loss Curves", 
                     xlabel="Epoch", ylabel="Loss", figsize=(10, 6), fname=None, subsample=100):
    """
    Plot multiple lists of loss values on the same graph.
    
    Args:
        loss_lists: List of lists, where each inner list contains loss values
        labels: List of strings for legend labels (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        fname: Filename to save the plot (without extension, saves as .jpg)
    """
    plt.figure(figsize=figsize)
    
    # Generate default labels if none provided
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(loss_lists))]
    
    # Plot each loss curve
    for i, losses in enumerate(loss_lists):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs[::subsample], losses[::subsample], marker='o', markersize=1, 
                linewidth=1, label=labels[i], alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Make it look nice
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save figure if filename provided
    if fname:
        plt.savefig(f"{fname}.jpg", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved as {fname}.jpg")
    
    return plt.gcf()

    
# Example usage with sample data
if __name__ == "__main__":
    # Sample loss data for different models/experiments
    loss_1 = [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28]
    loss_2 = [3.0, 2.1, 1.5, 1.0, 0.8, 0.6, 0.45, 0.38, 0.32, 0.29]
    loss_3 = [2.8, 1.9, 1.3, 0.95, 0.75, 0.55, 0.42, 0.36, 0.31, 0.27]
    
    # List of all loss curves
    all_losses = [loss_1, loss_2, loss_3]
    model_names = ["ResNet", "VGG", "DenseNet"]
    
    # Create the plot and save it
    fig = plot_loss_curves(all_losses, labels=model_names, 
                          title="Model Comparison - Training Loss",
                          fname="model_comparison_loss")
    
    plt.show()
    
    # Alternative: Plot with custom styling
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (losses, label) in enumerate(zip(all_losses, model_names)):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, 
                color=colors[i % len(colors)],
                marker='o', 
                markersize=6,
                linewidth=3,
                label=label,
                alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale can be helpful for loss curves
    
    # Styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Optional: Save this plot too
    # plt.savefig("custom_loss_plot.jpg", dpi=300, bbox_inches='tight')
    
    plt.show()

# Function for more advanced plotting with validation loss
def plot_train_val_losses(train_losses, val_losses=None, labels=None, fname=None):
    """
    Plot training and validation losses for multiple models.
    
    Args:
        train_losses: List of lists containing training losses
        val_losses: List of lists containing validation losses (optional)
        labels: List of model names
        fname: Filename to save the plot (without extension, saves as .jpg)
    """
    n_models = len(train_losses)
    
    if labels is None:
        labels = [f"Model {i+1}" for i in range(n_models)]
    
    plt.figure(figsize=(12, 6))
    
    for i in range(n_models):
        epochs = range(1, len(train_losses[i]) + 1)
        
        # Plot training loss
        plt.plot(epochs, train_losses[i], 
                label=f"{labels[i]} - Train", 
                linewidth=2, marker='o', markersize=4)
        
        # Plot validation loss if provided
        if val_losses and i < len(val_losses):
            plt.plot(epochs, val_losses[i], 
                    label=f"{labels[i]} - Val", 
                    linewidth=2, linestyle='--', marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure if filename provided
    if fname:
        plt.savefig(f"{fname}.jpg", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved as {fname}.jpg")
    
    plt.show()