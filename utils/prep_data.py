import cv2
from tqdm import tqdm
import glob
import os
import random

train_paths = glob.glob('data/train/videos/*')
os.makedirs('data/train/images', exist_ok=True)

test_paths = glob.glob('data/public_test/public/videos/*')
os.makedirs('data/public_test/public/images', exist_ok=True)

# # Extract frames from training data
# print('Extracting frames from training data')
# for train_path in tqdm(train_paths):
#     vidcap = cv2.VideoCapture(train_path)
#     length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if length > 11:
#         frames = random.sample(range(1, length), 10)
#         vidname = train_path.split('/')[-1].split('.')[0]
#         for frame_id in frames:
#             vidcap.set(1, frame_id)
#             success, image = vidcap.read()
#             if not success:
#                 print('Failed to read video {}'.format(vidname))
#                 break

#             cv2.imwrite('data/train/images/{}_{}.png'.format(vidname, frame_id), image)
        

# Extract frames from public test data
print('Extracting frames from public test data')
for test_path in tqdm(test_paths):
    vidcap = cv2.VideoCapture(test_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if length > 11:
        frames = random.sample(range(1, length), 10)
        vidname = test_path.split('/')[-1].split('.')[0]
        for frame_id in frames:
            vidcap.set(1, frame_id)
            success, image = vidcap.read()
            if not success:
                print('Failed to read video {}'.format(vidname))
                break

            cv2.imwrite('data/public_test/public/images/{}_{}.png'.format(vidname, frame_id), image)