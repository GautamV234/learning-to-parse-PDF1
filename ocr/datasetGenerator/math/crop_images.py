from PIL import Image
import os 
# Opens a image in RGB mode
images = os.listdir("images")
for image_path in images:
    os.chdir('C:\work\web scrape\images')
    im = Image.open(image_path)
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    # Setting the points for cropped image
    left = 5
    top = height / 4
    right = 164
    bottom = 3 * height / 4
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top+500, right+1500, bottom+120))
    print(f"image name is {image_path}")
    # Shows the image in image viewer
    # im1.show()
    os.chdir('C:\work\web scrape\cropped_images')
    im1.save(image_path)