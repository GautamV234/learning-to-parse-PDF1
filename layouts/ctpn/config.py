import os


icdar17_mlt_img_dir = 'data/mlt/image/'
icdar17_mlt_gt_dir = 'data/mlt/label/'
num_workers = 4  
pretrained_weights =  "checkpoints/epoch=9-step=3259.ckpt"
checkpoints_dir = "checkpoints/"


batch_size = 10
num_gpus = 1

max_epochs = 30  
anchor_scale = 16

IOU_NEGATIVE = 0.5 # 0.3
IOU_POSITIVE = 0.9 # 0.7
IOU_SELECT = 0.9 # 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

prob_thresh = 0.5
height = 720
