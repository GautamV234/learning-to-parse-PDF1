import cv2
import os
import glob
import numpy as np
os.makedirs('mixPrima/test/images/', exist_ok=True)
os.makedirs('mixPrima/test/annotations/', exist_ok=True)

os.makedirs('mixPrima/train/images/', exist_ok=True)
os.makedirs('mixPrima/train/annotations/', exist_ok=True)

latexpaths = glob.glob('latex/*.png')
annotationToTreat = 2

for s in ['test', 'train']:

    for path in glob.glob(f'prima/{s}/images/*.png'):
        filename = os.path.basename(path).split('.')[0]

        img = cv2.imread(path)
        img = cv2.resize(img, (1280, 1600), interpolation=cv2.INTER_NEAREST)
        print(path)
        annotation = cv2.imread(os.path.join(f'prima/{s}/annotations/' + filename + '.png'), 0)
        for i in range(np.random.randint(0, 10)):
            try:
                latexpath = np.random.choice(latexpaths)
                latex = cv2.imread(latexpath)
                latex = cv2.resize(latex, (0, 0), fx = 0.6, fy = 0.6, interpolation=cv2.INTER_NEAREST)

                imgmean = np.mean(img, axis = (0,1)).astype(np.uint8)
                latexgray = cv2.cvtColor(latex, cv2.COLOR_BGR2GRAY)

                latexgray[latexgray > 128] = 255
                latexgray[latexgray <= 128] = 0

                latexgray = cv2.bitwise_not(latexgray)

                # find coordinates where latexgray is white

                y, x = np.where(latexgray == 255)

                minx = np.min(x)
                miny = np.min(y)
                maxx = np.max(x)
                maxy = np.max(y)
                latexgray = latexgray[max(miny - 10, 0):min(maxy+ 10, img.shape[0]), max(0, minx - 10):min(maxx + 10, img.shape[1])]


                latex[latex < 128] = 0
                latex[latex >= 128] = 255

                latex = latex[max(miny - 10, 0):min(maxy+ 10, img.shape[0]), max(0, minx - 10):min(maxx + 10, img.shape[1])]

                latexgray =  255 - latexgray

                

                # convert to bgr

                # place the latex randomly in the image
                x = np.random.randint(0, img.shape[1] - latex.shape[1])
                y = np.random.randint(0, img.shape[0] - latex.shape[0])
                patch = img[y : y + latex.shape[0], x : x + latex.shape[1]].copy()

                # replace black in patch with image mean
                data = np.reshape(patch, (-1,3))
                data = np.float32(data)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                flags = cv2.KMEANS_RANDOM_CENTERS
                compactness,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)
                centers = sorted(centers, key = lambda x: np.sum(x), reverse = True)

                imgmean = centers[0].astype(np.uint8)
                latex[latexgray == 255] = imgmean
                img[y : y + latex.shape[0], x : x + latex.shape[1]] = latex

                annotation[y : y + latex.shape[0], x : x + latex.shape[1]] = annotationToTreat
            except:
                continue

        anno = annotation.copy().astype(np.float)
        anno/=anno.max()
        # cv2.imshow('annotation', (anno*255).astype(np.uint8))
        # cv2.imshow('asdf', img)

        # if cv2.waitKey(1) == ord('q'):
        #     exit() 

        cv2.imwrite(os.path.join(f'mixPrima/{s}/images/' + filename + '.png'), img)
        cv2.imwrite(os.path.join(f'mixPrima/{s}/annotations/' + filename + '.png'), annotation)

        
