import torch
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
import pandas as pd

from model.model import LD_Baseline
from utils.dataset import LD_test

def parse_args():
    parser = argparse.ArgumentParser(description='Inference arguments for LD baseline model')
    parser.add_argument('--data', '-d', type=str, default='data/public_test/public/images/', help='Path to folder containing test data')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/baseline.pth', help='Path to checkpoint')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Initialize model
    model = LD_Baseline()
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.cuda()
    model.eval()

    # Load data
    test_set = LD_test(args.data)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Testing loop
    outputs = torch.empty((1, 2), dtype=torch.int32, device=torch.device('cuda'))
    with tqdm(total=len(test_set), desc=f'Predicting', unit='img') as pbar:
        for batch in test_loader:
            images = batch['image'].cuda().to(torch.float32)
            vidnames = batch['vidname'].cuda().to(torch.int32)

            with torch.no_grad():
                pred = torch.round(model(images).squeeze(1)).to(torch.int32)
                stacks = torch.stack([vidnames, pred], dim=1)
                outputs = torch.cat((outputs, stacks), dim=0)

            pbar.update(images.shape[0])

    # Reduce mean
    outputs = outputs[1:, :].cpu().numpy()
    df = pd.DataFrame({'fname': outputs[:, 0], 'liveness_score': outputs[:, 1]})
    df = df.groupby('fname').mean().reset_index()
    df['fname'] = df['fname'].apply(lambda x: str(x) + '.mp4')
    df['liveness_score'] = df['liveness_score'].apply(lambda x: round(x))

    df.to_csv('predict.csv', index=False)