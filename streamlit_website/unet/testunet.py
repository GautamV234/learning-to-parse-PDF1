from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData


dm = SemanticSegmentationData.from_folders(
    predict_folder=r"testimgs/",
    transform_kwargs=dict(image_size=(800, 640)),
    batch_size = 1,
    )

model = SemanticSegmentation.load_from_checkpoint(r"trainedModel\unet\version_31_17042022\epoch=37-step=3268.ckpt")

from flash import Trainer

trainer = Trainer(max_epochs=1, gpus = 1)
predictions = trainer.predict(model, dm)
import cv2
import numpy as np
import torch



for i in range(len(predictions)):
    inp = (predictions[i][0]['input']).numpy()
    inp = np.transpose(inp, (1, 2, 0))
    pred = (predictions[i][0]['preds'])
    pred = torch.softmax(pred, dim = 0)
    pred = torch.argmax(pred, dim = 0).numpy()
    print(pred.shape, inp.shape)
    inp[:, :, 0][pred == 1] = 255
    # inp[:, :, 1][pred == 0] = 50
    pred = pred.astype(np.float)
    pred /= np.max(pred)

    cv2.imshow('image', inp)
    cv2.imshow('segm', pred)

    cv2.imwrite('outputs/' + str(i) + '.png', inp)
    cv2.waitKey(1)