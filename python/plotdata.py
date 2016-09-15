#Importing pyplot
from matplotlib import pyplot as plt
import numpy as np

def simple_fig():
    #Plotting to our canvas
    plt.plot([1,2,3],[4,5,1])

    #Showing what we plotted
    plt.show()

def display_rgb_image():
    rgb_image = np.zeros((100, 100, 3), dtype='uint8')
    rgb_image = np.random.rand(100, 100, 3)


    title_str = 'Test RGB image'
    plt.title(title_str)
    plt.ylabel('Y');
    plt.xlabel('X');
    plt.imshow(rgb_image)

def text_on_image():

    import Image, ImageDraw

    img = Image.new('RGB', (200, 100))
    d = ImageDraw.Draw(img)
    d.text((20, 20), 'Hello', fill=(255, 255, 255))

    img.save("textimg_hello.png")

if __name__ == "__main__":

    text_on_image()

    display_rgb_image()

    simple_fig()