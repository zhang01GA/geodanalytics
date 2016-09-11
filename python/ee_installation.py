""" For Linux ubuntu installation, follow the  guide: https://developers.google.com/earth-engine/python_install

     pin install  scikit-image
     apt install ffmpeg
     pip install google-api-python-client
     sudo pip install pyCrypto
     apt-get install libssl-dev
     sudo pip install 'pyOpenSSL>=0.11'
     sudo pip install earthengine-api

#   See https://developers.google.com/api-client-library/python/
"""

# Import the Earth Engine Python Package
import ee

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()

# Print the information for an image asset.
image = ee.Image('srtm90_v4')
print(image.getInfo())
