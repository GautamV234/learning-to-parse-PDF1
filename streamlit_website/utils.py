import cv2
import numpy as np


def postProcess(image, seg):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image[seg == 0] = 0
    img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    num_labels, labels = cv2.connectedComponents(img)


    for i in range(1, num_labels):
        
        y, x = np.where(labels == i)

        miny = np.min(y)
        maxy = np.max(y)
        minx = np.min(x)
        maxx = np.max(x)

        portion = img[miny:maxy, minx:maxx]

        columnsum = np.sum(portion, axis=0)

        splitpoints = np.where(columnsum == 0)
        prevsplitpoint = splitpoints[0]
        sentences = []

        for i in range(1, len(splitpoints[0])):
            sentences.append(portion[:, prevsplitpoint:splitpoints[0][i]])
            prevsplitpoint = splitpoints[0][i]


        



        return sentences

    


    return image