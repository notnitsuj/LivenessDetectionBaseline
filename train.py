import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import wandb
import numpy as np

from model.model import LD_Baseline
from utils.dataset import LD, split_data
from utils.utils import set_seed, get_args, EarlyStopper

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    # Logging
    wandb.init(project="Zalo22LivenessDetection", entity="notnitsuj", tags=["baseline"])
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "architecture": "resnet50"
    }

    # Load data
    train_split, val_split, labels = split_data(label_file=args.data + 'label.csv', val_frac=args.val)

    train_set = LD(img_path=args.data + 'images/', vid_list=train_split, labels=labels)
    val_set = LD(img_path=args.data + 'images/', vid_list=val_split, labels=labels)

    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # Model
    model = LD_Baseline(args.backbone).cuda()
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    early_stopper = EarlyStopper(patience=10, min_delta=0.1)

    # Training loop
    min_val_loss = np.inf
    for epoch in range(args.epochs):
        model.train()
        train_loss = []

        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].cuda().to(torch.float32)
                labels = batch['label'].cuda().to(torch.float32)

                pred = model(images).squeeze(1)
                loss = criterion(pred, labels) 
                train_loss.append(loss.item())

                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])

        # Evaluate after each epoch
        model.eval()
        val_loss = []

        with tqdm(total=len(val_set), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in val_loader:
                images = batch['image'].cuda().to(torch.float32)
                labels = batch['label'].cuda().to(torch.float32)

                with torch.no_grad():
                    pred = model(images).squeeze(1)
                    loss = criterion(pred, labels) 
                    val_loss.append(loss.item())

                pbar.update(images.shape[0])

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        scheduler.step(mean_val_loss)

        wandb.log({"train_loss": mean_train_loss, "val_loss": mean_val_loss})

        # Check for early stopping criterion
        if early_stopper.early_stop(mean_val_loss):             
            break

        # Save checkpoint
        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            torch.save(model.state_dict(), "checkpoints/baseline.pth")