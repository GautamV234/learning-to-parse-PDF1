import numpy as np
import glob
import cv2
import os
import random
import shutil

os.makedirs('prima/test/images/', exist_ok=True)
os.makedirs('prima/test/annotations/', exist_ok=True)

os.makedirs('prima/train/images/', exist_ok=True)
os.makedirs('prima/train/annotations/', exist_ok=True)




print('Total images = ', len(glob.glob(r'final_images/images/*.png')))

paths = glob.glob(r'final_images/images/*.png')
random.seed(42)
random.shuffle(paths)

splitratio = 0.1

test_paths = paths[:int(len(paths) * splitratio)]
train_paths = paths[int(len(paths) * splitratio):]
print('Generating test')
for path in test_paths:

    filename = path.split('/')[-1].split('.')[0].split('\\')[-1]

    shutil.copyfile(path, r'prima/test/images/' + filename + '.png')
    shutil.copyfile(r'final_images/annotations/' + filename + '.png', r'prima/test/annotations/' + filename + '.png')

print('Generating train')
for path in train_paths:

    filename = path.split('/')[-1].split('.')[0].split('\\')[-1]

    shutil.copyfile(path, r'prima/train/images/' + filename + '.png')
    shutil.copyfile(r'final_images/annotations/' + filename + '.png', r'prima/train/annotations/' + filename + '.png')
