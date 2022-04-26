import xml.etree.ElementTree as ET
import numpy as np
import glob
import cv2
import os
os.makedirs('final_images/images/', exist_ok=True)
os.makedirs('final_images/annotations/', exist_ok=True)
os.makedirs('final_images/viz/', exist_ok=True)
for path in glob.glob(r'XML/*.xml'):
    tree = ET.parse(path)
    root = tree.getroot()
    for child in root:
        if "Page" in child.tag:
            filename = child.attrib['imageFilename'].split('.')[0]

            print(filename, 'Dims:',int(child.attrib['imageHeight']), int(child.attrib['imageWidth']) )
            try:
                img = np.zeros((int(child.attrib['imageHeight']), int(child.attrib['imageWidth'])), dtype=np.uint8)
                for region in child:
                    if 'TextRegion' in region.tag:
                        coords = region[0]
                        coordslist = []
                        for coord in coords:
                            coord = coord.attrib
                            coordslist.append([int(coord['x']), int(coord['y'])])
                        cv2.fillPoly(img, np.array([coordslist]), 1, )

            except:
                print('FAILL!')
                pass

            break
    

    img = cv2.resize(img, (1280, 1600), interpolation= cv2.INTER_NEAREST)
    try:
        cv2.imwrite(r'final_images/annotations/' + filename + '.png',   img)

        
        imm = cv2.imread(r'Images/' + filename + '.tif')
        imm = cv2.resize(imm, (1280, 1600), interpolation= cv2.INTER_NEAREST)
        cv2.imwrite(r'final_images/images/' + filename + '.png', imm)
    except:
        pass