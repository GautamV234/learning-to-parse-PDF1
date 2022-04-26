from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dm = SemanticSegmentationData.from_folders(
    train_folder="/home/varun/trans/unet/data/prima/train/images",
    train_target_folder="/home/varun/trans/unet/data/prima/train/annotations",
    val_split=0.05,
    batch_size = 2,
    transform_kwargs=dict(image_size=(1600, 1280)),
    num_classes=2,
    num_workers = 4,
)

model = SemanticSegmentation(
  head="unet", 
  backbone='efficientnet-b0', 
  num_classes=dm.num_classes  
)

from flash import Trainer

trainer = Trainer(max_epochs=50, gpus = 1)
trainer.fit(model, datamodule=dm)
trainer.save_checkpoint("mixPrimaDual.pt")