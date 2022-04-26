from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData


dm = SemanticSegmentationData.from_folders(
    predict_folder=r"testimgs/",
    transform_kwargs=dict(image_size=(1600, 1280)),
    batch_size = 1,
    )


from flash import Trainer

import cv2
import numpy as np
import torch
# model = SemanticSegmentation.load_from_checkpoint(r"semantic_segmentation_model.pt")
# model = SemanticSegmentation.load_from_checkpoint(r"lightning_logs/version_42_mixprimadual/checkpoints/epoch=49-step=9599.ckpt")
# model = SemanticSegmentation.load_from_checkpoint(r"lightning_logs/version_41_mixprimatern/checkpoints/epoch=49-step=9599.ckpt")
model = SemanticSegmentation.load_from_checkpoint(r"lightning_logs/version_43_plainprima/checkpoints/epoch=49-step=9599.ckpt")


trainer = Trainer(max_epochs=1, gpus = 1)
predictions = trainer.predict(model, dm)


for i in range(len(predictions)):
    inp = (predictions[i][0]['input']).numpy()
    inp = np.transpose(inp, (1, 2, 0))/np.max(inp)
    pred = (predictions[i][0]['preds'])
    pred = torch.softmax(pred, dim = 0)
    pred = torch.argmax(pred, dim = 0).numpy()
    # print(pred.shape, inp.shape)
    inp[:, :, 0][pred == 1] = 255
    inp[:, :, 1][pred == 2] = 255
    # inp[:, :, 1][pred == 0] = 50    pred = pred.astype(np.float)

    cv2.imwrite('segs/' + str(i) + '.png', (pred*255).astype(np.uint8))
    cv2.imwrite('outputs/' + str(i) + '.png', (inp*255).astype(np.uint8))
    # pred /= np.max(pred)
