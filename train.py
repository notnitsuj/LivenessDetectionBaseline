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
from utils.metrics import compute_eer

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    # Logging
    wandb.init(project="Zalo22LivenessDetection", entity="notnitsuj", tags=["baseline"])
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "architecture": args.backbone,
        "img_size": args.size
    }

    # Load data
    train_split, val_split, labels = split_data(label_file=args.data + 'label.csv', val_frac=args.val)

    train_set = LD(img_path=args.data + 'images/', vid_list=train_split, labels=labels, newsize=[args.size, args.size])
    val_set = LD(img_path=args.data + 'images/', vid_list=val_split, labels=labels, newsize=[args.size, args.size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size*2, num_workers=4, pin_memory=True)

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
        outputs = torch.empty((1, 3), dtype=torch.int32, device=torch.device('cuda'))

        with tqdm(total=len(val_set), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in val_loader:
                images = batch['image'].cuda().to(torch.float32)
                labels = batch['label'].cuda().to(torch.float32)
                vidname = batch['vidname'].cuda().to(torch.float32)

                with torch.no_grad():
                    pred = model(images).squeeze(1)
                    loss = criterion(pred, labels) 
                    val_loss.append(loss.item())

                    stacks = torch.stack([vidname, labels, pred], dim=1)
                    outputs = torch.cat((outputs, stacks), dim=0)

                pbar.update(images.shape[0])


        # Calculate mean loss
        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        scheduler.step(mean_val_loss)

        # Calculate EER metrics
        # Ref: https://stackoverflow.com/a/66871328
        outputs = outputs[1:, :].numpy(force=True)
        groups = outputs[:, 0].copy()
        outputs = np.delete(outputs, 0, axis=1)

        _ndx = np.argsort(groups)
        _, _pos, g_count  = np.unique(groups[_ndx], 
                                        return_index=True, 
                                        return_counts=True)

        g_sum = np.add.reduceat(outputs[_ndx], _pos, axis=0)
        g_mean = g_sum / g_count[:, None]

        eer = compute_eer(g_mean[:, 0].astype(np.uint8).tolist(), g_mean[:, 0].tolist())

        # Log
        wandb.log({"train_loss": mean_train_loss, "val_loss": mean_val_loss, "eer": eer})
        print('Train loss: {}'.format(mean_train_loss))
        print('Val loss: {}'.format(mean_val_loss))
        print('EER: {}'.format(eer))

        # Check for early stopping criterion
        if early_stopper.early_stop(mean_val_loss):             
            break

        # Save checkpoint
        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            torch.save(model.state_dict(), "checkpoints/{}.pth".format(args.backbone))