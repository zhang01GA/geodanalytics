#Importing pyplot
from matplotlib import pyplot as plt
import numpy as np
import Image, ImageDraw

from PIL import ImageFont
import os, sys
import pathlib



def add_text_onto_image(p2imgfile, display=None):
    """
    Added text to an existing image file; 
    create a new image file 
    p2imgfile: path 2o an image file
    """

    pf= pathlib.Path(p2imgfile)
    
    filen=pf.name
    pathn=pf.parent
    
    
    img = Image.open(p2imgfile)
    
    xsize, ysize=img.size
    
    d = ImageDraw.Draw(img)
    
    textstr= filen[:-4]
    
    fsize=36
    
    myfont = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", fsize)
    
    d.text((0, 0), textstr, fill=(255, 255, 255), font=myfont) #white

    #d.text((0, 0), textstr, fill=(0,0,0), font=myfont)    
    
    newfn='text_'+filen
    
    path2newfile=str(pathn.joinpath(newfn))

    img.save(path2newfile)
    
    # want to display this image?
    if display is not None:
        img.show()
    else:
        pass
    
#############################################################    
if __name__ == "__main__":
    
    #imgf= '/short/v10/fxz547/Dexport/LakeGeorge_ls8_NBAR_2014-08-12.png'
    
    if len(sys.argv) <2:
        print("Must provide one or more image files ")
        sys.exit(1)
    elif len(sys.argv) == 2:
        add_text_onto_image(sys.argv[1], display=True)
    else:
        for imgf in sys.argv[1:]:
            print ("doing  ", imgf)
    
            add_text_onto_image(imgf)

#==============================================================================
#     text_on_image()
# 
#     display_rgb_image()
# 
#     simple_fig()
#==============================================================================
