import numpy as np
import cv2
import glob
for path in glob.glob('prima/test/annotations/*.png'):
    img = cv2.imread(path).astype(np.float)
    img /= 2
    img *= 255
    cv2.imshow(path, img.astype(np.uint8))
    if cv2.waitKey(0) == ord('q'):
        exit()
    cv2.imwrite('viz.png', img.astype(np.uint8))
    cv2.destroyAllWindows()


