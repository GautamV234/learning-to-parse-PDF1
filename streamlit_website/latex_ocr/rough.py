from PIL import Image
from cli import run_pix2tex

img = Image.open('test_images/img1.jpg')

run_pix2tex(img)