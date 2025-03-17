import os
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.data import Dataset, random_split, SubsetRandomSampler, WeightedRandomSampler

# CUDA_VISIBLE_DEVICES=3  torchrun --standalone --nproc-per-node=1   ddp.py
# CUDA_VISIBLE_DEVICES=3,5  torchrun --standalone --nproc-per-node=2   ddp.py

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --standalone scripts/train.py +experiments=deft-voice-423 minibatch_size=1 targets_on_the_fly=False accum_iters=2 val_freq=1000 save_freq=1000 prefetch_factor=null num_workers=0
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --standalone ddp.py
# python -m torch.distributed.run ddp.py 
# torchrun --standalone --nnodes=1 --nproc-per-node=1 ddp.py 
    
try:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    IS_HEAD_NODE = LOCAL_RANK == 0
except Exception:
    raise Exception("launch script using torchrun")


######################
# 1) Define the model
######################
class SimpleNet(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, hidden=4000, out_dim=10, num_inner=20):
        """
        SimpleNet is a simple feedforward neural network with hidden layers.
        Default parameters are selected to max out 10GB 3070 GPU memory usage.
        in_dim: input dimension
        hidden: hidden dimension
        out_dim: output dimension
        num_inner: number of inner layers
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_inner = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.num_inner = num_inner

    def forward(self, x):
        # Flatten input [B, 3, 32, 32] -> [B, 3*32*32]
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        for _ in range(self.num_inner):
            x = self.fc_inner(x)
            x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


##############################
# 2) Training loop (per rank)
##############################
def ddp_train_loop(rank, world_size, args):
    """
    rank: process rank [0..world_size-1]
    world_size: total number of DDP processes
    args: dictionary with config (backend, epochs, save_path, etc.)
    """
    
    # a) Create a FileStore and init PG (no TCP or NCCL)
    store_path = os.path.join(args["save_path"], "filestore")
    os.makedirs(store_path, exist_ok=True)

    if False:
        # The second argument to FileStore is the number of processes (world_size)
        store = dist.FileStore(os.path.join(store_path, "store"), WORLD_SIZE)

        dist.init_process_group(
            backend=args["backend"],  # "gloo"
            store=store,
            world_size=WORLD_SIZE,
            rank=LOCAL_RANK
        )
        device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
        # b) Create local model & DDP wrapper
        model = SimpleNet()
        model.to(device)
        ddp_model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    else:
        # store = dist.TCPStore("localhost", 29500, WORLD_SIZE, is_master=(rank == 0)) # TBD
        # set device
        assert torch.cuda.is_available(), "assumes GPU"
        dist.init_process_group(
            backend=dist.Backend.NCCL,
            # store=store, # TBD:  assert rank >= 0, "rank must be non-negative if using store"
            world_size=WORLD_SIZE,
            # rank=rank, # RuntimeError: The server socket has failed to listen on any local network address.
            )
        print(f" *** ddp rank={rank}  dist.get_rank()={dist.get_rank()}") # *** ddp rank=0  dist.get_rank()=3
        rank = dist.get_rank()

        print(f"Start running basic DDP example on rank {rank}.")

        device = torch.device("cuda", index=LOCAL_RANK)
        torch.cuda.set_device(device)

        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()    
        # set device
        print(f"device={device} device_id={device_id} torch.cuda.device_count()={torch.cuda.device_count()}")
        #torch.cuda.set_device(device)


        # enabled cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        # wrap model for DDP
        # initialize model
        model = SimpleNet()
        model = model.to(device)
        ddp_model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)



    # c) Create dataset / dataloader
    train_set = CIFAR10(root=args["data_dir"], train=True, download=True, transform=ToTensor())
    
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)       
    train_idx = list(range(32000)) # max 50K
    # sampler = SubsetRandomSampler(train_idx)

    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2, prefetch_factor=2, persistent_workers=True)
    

    # d) Setup optimizer, criterion
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # e) Training
    time_start = time.time()
    print(f"args[epochs]={args['epochs']}")
    for epoch in range(args["epochs"]):
        # sampler.set_epoch(epoch)
        ddp_model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0: #  and rank == 0:
                print(f"[Rank {LOCAL_RANK}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # f) If rank=0, save checkpoint
        if LOCAL_RANK == 0:
            state_dict = model.state_dict()
            ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")
            torch.save(state_dict, ckpt_path)
            print(f"[Rank {LOCAL_RANK}] Saved {ckpt_path}")

    print(f"end training rank={LOCAL_RANK}: time={int(1000*(time.time() - time_start)/args['epochs'])}")

    dist.destroy_process_group()
    print(f"[Rank {LOCAL_RANK}] training complete.")


##########################
# 3) CPU test process
##########################
def cpu_test_proc(args):
    """
    Runs on CPU and checks for each epoch exactly once,
    then exits when done (so we don't get stuck waiting).
    """
    test_set = CIFAR10(root=args["data_dir"], train=False, download=True, transform=ToTensor())
    test_loader2 = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=1, prefetch_factor=2, persistent_workers=True)
    test_len = len(test_loader2) // args['num_processes']
    test_start = args['iproc'] * test_len
    test_end = test_start + test_len
    #test_loader = test_loader[test_start:test_end]
    # Define indices for the subset
    subset_indices = list(range(test_start, test_end)) # [1, 4, 7, 10, 13, 16, 19]    
    # Create a Subset
    subset = Subset(test_set, subset_indices)
    test_loader = DataLoader(subset, batch_size=2)
    print(f"***[Test Process] Started, using CPU args={args}")
    print(f"*** test_len={test_len} test_start={test_start} test_end={test_end}")
    criterion = nn.CrossEntropyLoss()

    # We'll do exactly 'args["epochs"]' test checks
    for epoch in range(args["epochs"]):
        ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")

        # Wait until the checkpoint for this epoch appears
        while not os.path.exists(ckpt_path):
            time.sleep(5)  # Keep checking every 5 seconds

        # Once found, load and evaluate
        model = SimpleNet()
        # FutureWarning: if you want to avoid full pickle loading, use 'weights_only=True'
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        print(f"*** [Test Process {args['iproc']},test_start={test_start},test_end={test_end},test_len={test_len}: {ckpt_path}] Epoch {epoch}, Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

    print("[Test Process] All epochs tested. Exiting.")


##########################
# 4) Main entry point
##########################
args = {
    # Force "gloo" if "nccl" when not compiled on Windows
    "backend": "nccl", # "gloo",
    "epochs": 10,
    "data_dir": "./data",
    "save_path": "./checkpoints",
    'num_processes': 2,
}
def main():

    os.makedirs(args["save_path"], exist_ok=True)    

    if True:
        files = glob.glob('./checkpoints/*')
        for f in files:
            print(f"removing {f}")
            # Check if the file exists before deleting
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"e={e}")

    processes = []
    for iproc in range(args['num_processes']):
        # Remove daemon=True so the test process can spawn DataLoader workers
        args['iproc'] = iproc
        test_process = mp.Process(target=cpu_test_proc, args=(args,))
        test_process.start()
        processes.append(test_process)

    mp.spawn(
        fn=ddp_train_loop,
        args=(WORLD_SIZE, args),
        nprocs=1,
        join=True
    )

    # test_process.join()
    for proc in processes:
        proc.join()
        print("[Main] All processes finished.")


if __name__ == "__main__":
    print("starting spawn")    
    time_start = time.time()
    print("Usage: CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --standalone ddp.py <ddp>")
    if len(sys.argv) == 1:
        # GPU + CPU test process
        mp.set_start_method("spawn", force=True)
        main()
    else:
        # no CPU test process
        os.makedirs(args["save_path"], exist_ok=True)    
        ddp_train_loop(LOCAL_RANK, WORLD_SIZE, args)
    print(f"end spawn time={int(1000*(time.time() - time_start))}")

