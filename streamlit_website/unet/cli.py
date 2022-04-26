import numpy as np

from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData



from flash import Trainer

import cv2
import numpy as np
import torch

def runUnet(img):
    # img = np.squeeze(img)
    arr = np.array(img).astype(float)
    arr = np.transpose(arr, (1, 2, 0))
    

    arr /= 255.0

    
    dm = SemanticSegmentationData.from_numpy(
        # predict_folder=r"testimgs/",
        predict_data=[arr],
        transform_kwargs=dict(image_size=(800, 640)),
        batch_size = 1,
        )
    model = SemanticSegmentation.load_from_checkpoint(r"E:\Google Drive\Acads\Notes\final sem\ML\learning-to-parse-PDF\streamlit_website\unet\epoch=49-step=9599.ckpt")


    trainer = Trainer(max_epochs=1,gpus=1)
    predictions = trainer.predict(model, dm)
    
    for i in range(len(predictions)):
        inp = (predictions[i][0]['input']).numpy()  
        inp = np.transpose(inp, (1, 2, 0))/np.max(inp)
        pred = (predictions[i][0]['preds'])
        pred = torch.softmax(pred, dim = 0)
        pred = torch.argmax(pred, dim = 0).numpy()
        inp[:, :, 0][pred == 1] = 255
        inp[:, :, 1][pred == 2] = 255

    torch.cuda.empty_cache()



    return inp, pred



