#!/usr/bin/env python

###############################################################################
# 
# Purpose:  Create a browse image with 1000 pixels wide from RGB TIF input files.
# Paul G:   inclusion of basig FAST format handling
# 
# Date:     2012-06-01
# Author:   Simon.Oliver@ga.gov.au and  Fei.Zhang@ga.gov.au
# Revisions:
#   2012-07-01: Fei Zhang refactored the program module into a class
#           and fixed all tab-space mixed issues
#   2012-07-05: Fei and Paul make the create() method to take either
#           "LSTRETCH" or "CONVERT" parameters, to choose an available jpeg creation algorithm
#           The LSTRETCH is fast and make better images. But it requires GDAL and numpy installed with the python
#  2012-08-06: Paul: Inclusion of FAST format code from GISTA packaging
#  2010-08-07: Paul and Simon O: Merged FAST format handling into main GDAL code utalizing
#             same stretching algorithm
#
#
###############################################################################

import logging
import sys
import subprocess
import math as math
import os
import tempfile

_log = logging.getLogger(__name__)

class BrowseImgCreator(object):
    def __init__(self):

        self.incols = None
        self.inrows = None
        self.inpixelx = None

        pass

    def setInput_RGBfiles(self, rfile, gfile, bfile, nodata_value ):
        self.red_file = rfile
        self.green_file = gfile
        self.blue_file = bfile
        self.nodata = nodata_value

        return

    def setOut_Browsefile(self, browsefile, cols):
        self.outthumb = browsefile
        self.outcols = cols
        self.tempDir = os.path.dirname(browsefile)

        return

    def create(self, alg):
        # next "Path" or North Up from the MTL file
        baseDir = os.path.dirname(self.green_file)
        # print "GREEN Band:", baseDir, self.green_file
        for root, dirs, files in os.walk(baseDir):
            for f in files:
                if f.find('_MTL.txt') > 1:
                    mtl_header_path = os.path.join(baseDir, f)
        fp = open(mtl_header_path)
        input_data = fp.read()
        for line in input_data.split('\n'):
            if line.find('ORIENTATION') > -1:
                self.orientation = line.split('=')[1].strip().strip('"')
            if line.find('PRODUCT_SAMPLES_REF') > -1:
                self.incols = int(line.split('=')[1].strip())
            if line.find('PRODUCT_LINES_REF') > -1:
                self.inrows = int(line.split('=')[1].strip())
            if line.find('GRID_CELL_SIZE_REF') > -1:
                self.inpixelx = float(line.split('=')[1].strip())

            # Diferent tags for Landsat-8:
            if line.find('REFLECTIVE_SAMPLES') > -1:
                self.incols = int(line.split('=')[1].strip())
            if line.find('REFLECTIVE_LINES') > -1:
                self.inrows = int(line.split('=')[1].strip())
            if line.find('GRID_CELL_SIZE_REFLECTIVE') > -1:
                self.inpixelx = float(line.split('=')[1].strip())

        # Special case for TIRS data set: THERMAL only
        if self.incols is None:
            for line in input_data.split('\n'):
                if line.find('THERMAL_SAMPLES') > -1:
                    self.incols = int(line.split('=')[1].strip())
                if line.find('THERMAL_LINES') > -1:
                    self.inrows = int(line.split('=')[1].strip())
                if line.find('GRID_CELL_SIZE_THERMAL') > -1:
                    self.inpixelx = float(line.split('=')[1].strip())


        _log.info("Orientation: %s", self.orientation)
        _log.info("Lines/pixels: %s, %s", self.inrows, self.incols)
        _log.info("Pixel Size: %s", self.inpixelx)

        if alg == "LSTRETCH":
            if self.orientation == 'NOM':
                return self.generate_Path_browse_image()
            else:
                return self.create_linear_stretch()

        else:
            _log.error("Unrecognised algorithm '%s'", alg)
            return -1


    def initial_fast(self, green_file):
        # locates and returns the FAST header file
        # along with RGB band combination

        # need to locate the HRF fast format header
        baseDir = os.path.dirname(self.green_file)
        fast_header_path = None
        for root, dirs, files in os.walk(baseDir):
            for f in files:
                if f.find('_HRF.FST') > 1:
                    fast_header_path = os.path.join(baseDir, f)

        if not fast_header_path:
            raise Exception("No fast header path found in dir %s" % baseDir)

        # must determine correct band numbers from "BANDS PRESENT"
        IredB = self.red_file.split('_B')[-1][0]
        IgreenB = self.green_file.split('_B')[-1][0]
        IblueB = self.blue_file.split('_B')[-1][0]
        p = subprocess.Popen(['grep', 'PRESENT', fast_header_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout_msg, stderr_msg) = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("Failure reading BANDS from file '%s'.  Exit status %d: %s" %
                               (fast_header_path, p.returncode, stderr_msg))

        bandsPresent = stdout_msg.strip().split('=')[1]
        for i in range(len(bandsPresent)):
            if bandsPresent[i] == IredB:
                redB = i + 1
            if bandsPresent[i] == IgreenB:
                greenB = i + 1
            if bandsPresent[i] == IblueB:
                blueB = i + 1
        return fast_header_path, redB, greenB, blueB

    ###############################################################################################

    def create_linear_stretch(self):

        # Linear contrast stretch using GDAL 
        #from osgeo import gdal
        import gdal
        import gdalconst
        import numpy.ma as ma

        _log.info("Generating Browse Image for Ortho/Map product")

        tempDir = self.tempDir
        _log.info("Temp directory: %s", tempDir)

        # working files

        tmpPrefix = "temp_" + os.path.basename(self.blue_file).split('_B')[0]
        file_to = os.path.join(tempDir, tmpPrefix + "_RGB.vrt")
        warp_to_file = os.path.join(tempDir, tmpPrefix + "_RGBwarped.vrt")
        outtif = os.path.join(tempDir, tmpPrefix + "_browseimg.tif")

        # Different initial step for FAST vs TIFF
        if self.green_file.find("FST") > 1:

            # fd, file_to_R = tempfile.mkstemp(suffix="_BandR.vrt")
            # fd, file_to_G = tempfile.mkstemp(suffix="_BandG.vrt")
            # fd, file_to_B = tempfile.mkstemp(suffix="_BandB.vrt")

            file_to_R = os.path.join(tempDir, tmpPrefix + "_band_R.vrt")
            file_to_G = os.path.join(tempDir, tmpPrefix + "_band_G.vrt")
            file_to_B = os.path.join(tempDir, tmpPrefix + "_band_B.vrt")

            (fast_header_path, redB, greenB, blueB) = self.initial_fast(self.green_file)

            _log.info("Processing red band")
            args = ['gdal_translate', '-b', str(redB), fast_header_path, file_to_R]
            subprocess.call(args)
            _log.info("... green band")
            args = ['gdal_translate', '-b', str(greenB), fast_header_path, file_to_G]
            subprocess.call(args)
            _log.info("... and blue band")
            args = ['gdal_translate', '-b', str(blueB), fast_header_path, file_to_B]
            subprocess.call(args)

            # Build the RGB Virtual Raster at full resolution
            step1 = [
                "gdalbuildvrt",
                "-overwrite",
                "-separate",
                file_to,
                file_to_R,
                file_to_G,
                file_to_B]

            _log.info("First step: %s", step1)

            subprocess.call(
                ["gdalbuildvrt",
                 "-overwrite",
                 "-separate",
                 file_to,
                 file_to_R,
                 file_to_G,
                 file_to_B], cwd = tempDir)

        else:
            # Build the RGB Virtual Raster at full resolution
            step1 = [
                "gdalbuildvrt",
                "-overwrite",
                "-separate",
                file_to,
                self.red_file,
                self.green_file,
                self.blue_file]

            _log.info("First step: %s", step1)
            subprocess.call(
                ["gdalbuildvrt",
                 "-overwrite",
                 "-separate",
                 file_to,
                 self.red_file,
                 self.green_file,
                 self.blue_file], cwd = tempDir)

        if not os.path.isfile(file_to):
            _log.error("error creating .vrt file '%s'", file_to)
            return -9

        # Determine the pixel scaling to get the outcols (1024 usually) wide thumbnail
        vrt = gdal.Open(file_to)
        intransform = vrt.GetGeoTransform()
        inpixelx = intransform[1]
        inrows = vrt.RasterYSize
        incols = vrt.RasterXSize
        _log.info("incols %s, inrows %s", incols, inrows)
        outcols = self.outcols
        if outcols == 0:
            # ZERO indicates full size so just convert
            outresx = inpixelx
            outrows = inrows
            outcols = incols
        else:
            outresx = inpixelx * incols / self.outcols  # output peg reolution
            outrows = int(math.ceil((float(inrows) / float(incols)) * self.outcols))
        _log.info("pixels: %s,%s,%s", outresx, outcols, outrows)

        subprocess.call(["gdalwarp",
                         "-of",
                         "VRT",
                         "-tr",
                         str(outresx),
                         str(outresx),
                         "-r",
                         "near",
                         "-overwrite",
                         file_to, warp_to_file])

        # Open VRT file to array
        vrt = gdal.Open(warp_to_file)
        bands = (1, 2, 3)

        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(outtif, outcols, outrows, 3, gdalconst.GDT_Byte)

        # Loop through bands and apply Scale and Offset
        for bandnum, band in enumerate(bands):
            vrtband = vrt.GetRasterBand(band)

            vrtband_array = vrtband.ReadAsArray()
            nbits = gdal.GetDataTypeSize(vrtband.DataType)
            dataTypeName = gdal.GetDataTypeName(vrtband.DataType)

            _log.info("nbits = %d, type = %s, self.nodata = %d" % (nbits, dataTypeName, self.nodata))

            dfScaleDstMin, dfScaleDstMax = 0.0, 255.0
            # Determine scale limits
            #dfScaleSrcMin = dfBandMean - 2.58*(dfBandStdDev)
            #dfScaleSrcMax = dfBandMean + 2.58*(dfBandStdDev)
            if dataTypeName == "Int16":   # signed integer
                count = 32767 + int(self.nodata)   #for 16bits 32767-999= 31768
                histogram = vrtband.GetHistogram(-32767, 32767, 65536)
            elif dataTypeName == "UInt16":  # unsigned integer
                count = 0
                histogram = vrtband.GetHistogram(-0.5, 65535.5, 65536)
            else:
                count = 0
                histogram = vrtband.GetHistogram()

            total = 0

            cliplower = int(0.01 * (sum(histogram) - histogram[count]))
            clipupper = int(0.99 * (sum(histogram) - histogram[count]))

            # print "count = ", count
            # print "len(histogram)", len(histogram)

            dfScaleSrcMin = count
            while total < cliplower and count < len(histogram) - 1:
                count =  count + 1
                total = total + int(histogram[count])
                dfScaleSrcMin = count

            if dataTypeName == "Int16":
                count = 32767 + int(self.nodata)
            else:
                count = 0
            total = 0
            dfScaleSrcMax = count
            while total < clipupper and count < len(histogram) - 1:
                count = count + 1
                total = total + int(histogram[count])
                dfScaleSrcMax = count

            if dataTypeName == "Int16":
                dfScaleSrcMin = dfScaleSrcMin - 32768
                dfScaleSrcMax = dfScaleSrcMax - 32768

            # GEMDOPS-1040 need to trap possible divide by zero in the stats
            # Check for Src Min == Max: would give divide by zero:
            srcDiff = dfScaleSrcMax - dfScaleSrcMin
            if srcDiff == 0:
                _log.warn("dfScaleSrc Min and Max are equal! Applying correction")
                srcDiff = 1

            # Determine gain and offset
            # dfScale = (dfScaleDstMax - dfScaleDstMin) / (dfScaleSrcMax - dfScaleSrcMin)
            dfScale = (dfScaleDstMax - dfScaleDstMin) / srcDiff
            dfOffset = -1 * dfScaleSrcMin * dfScale + dfScaleDstMin

            #Apply gain and offset    
            outdataset.GetRasterBand(band).WriteArray(
                (ma.masked_less_equal(vrtband_array, int(self.nodata)) * dfScale) + dfOffset)

            pass  # for loop

        outdataset = None
        vrt = None  # do this is necessary to allow close the files and remove warp_to_file below

        # GDAL Create doesn't support JPEG so we need to make a copy of the GeoTIFF
        subprocess.call(["gdal_translate", "-of", "JPEG", outtif, self.outthumb])

        # Cleanup working VRT files
        os.remove(file_to)
        os.remove(warp_to_file)
        os.remove(outtif)
        if self.green_file.find("FST") > 1:
            os.remove(file_to_R)
            os.remove(file_to_G)
            os.remove(file_to_B)
            # Done

        return outresx  # output jpeg resolution in meters

    def generate_Path_browse_image(self):
        """Uses GDAL to create a JPEG browse image for a Path dataset."""

        _log.info("Generating Browse Image for Path product")

        # calculate scale factor. required for TIF and FAST
        outcols = self.outcols
        if outcols == 0:
        # ZERO indicates full size so just convert
            outresx = self.inpixelx
            outrows = self.inrows
            outcols = self.incols
        else:
            outresx = self.inpixelx * self.incols / self.outcols  # output peg reolution
            outrows = int(math.ceil((float(self.inrows) / float(self.incols)) * self.outcols))
        _log.info("pixels: %s,%s,%s", outresx, outcols, outrows)

        # Different initial step for FAST vs TIFF
        if self.green_file.find("FST") > 1:

            (fast_header_path, redB, greenB, blueB) = self.initial_fast(self.green_file)

            args = ['gdal_translate',
                    '-b', str(redB),
                    '-b', str(greenB),
                    '-b', str(blueB),
                    '-outsize', str(outcols), str(outrows), '-of', 'JPEG', '-scale',
                    fast_header_path, self.outthumb]

            subprocess.call(args)
            gdal_xml_file = self.outthumb + '.aux.xml'
            if os.path.exists(gdal_xml_file):
                os.remove(gdal_xml_file)

            return outresx

        fd, outtif = tempfile.mkstemp(suffix="_browseimg.tif")
        args = ['gdal_merge.py', '-seperate', self.blue_file, self.green_file, self.red_file,
                '-o', outtif]
        _log.info("Running: '%s'", args)
        subprocess.call(args)

        # convert tiff to jpeg
        args = ['gdal_translate', '-b', '1', '-b', '2', '-b', '3',
                '-outsize', str(outcols), str(outrows), '-of', 'JPEG', '-scale',
                outtif, self.outthumb]
        _log.info("Running: '%s'", args)
        subprocess.call(args)

        os.path.remove(outtif)
        gdal_xml_file = self.outthumb + '.aux.xml'
        if os.path.exists(gdal_xml_file):
            os.remove(gdal_xml_file)

        return outresx


def createThumbnail(red, green, blue, nodata_val, outBrowseFile, width=0):
    bimgObj = BrowseImgCreator()
    bimgObj.setInput_RGBfiles(red, green, blue, nodata_val)
    bimgObj.setOut_Browsefile(outBrowseFile, int(width))
    resolution = bimgObj.create("LSTRETCH")
    return resolution


#############################################################################################
# Example Usage:
# python browseimg_creator.py 
#    /home/fzhang/SoftLabs/geolab/data/LandsatImages/LT5_20080310_091_077/L5091077_07720080310_B70.TIF 
#    /home/fzhang/SoftLabs/geolab/data/LandsatImages/LT5_20080310_091_077/L5091077_07720080310_B40.TIF 
#    /home/fzhang/SoftLabs/geolab/data/LandsatImages/LT5_20080310_091_077/L5091077_07720080310_B10.TIF 
#     0   LT5_20080310_091_077_test.jpg 1024
# nodata=0 for 8-bits images
# nodata=-999 for 16-bits images
############################################################################################        
if __name__ == "__main__":
# check for correct usage - if not prompt user

    print "Browse Image Creation using GDAL"
    if len(sys.argv) < 6:
        print "*----------------------------------------------------------------*"
        print ""
        print " thumbnail.py computes a linear stretch and applies to input image"
        print ""
        print "*----------------------------------------------------------------*"
        print ""
        print " usage: thumbnail.py <red image file> <green image file>"
        print " <blue image file> <output image file> <input null value>"
        print " <optional output width in pixels>"
        print "NOTE output width = 0 indicates full size browse image"
        sys.exit(1)

    jpeg_pixel_size = createThumbnail(sys.argv[1],
                                      sys.argv[2],
                                      sys.argv[3],
                                      sys.argv[4],
                                      sys.argv[5],
                                      0 if len(sys.argv) < 7 else int(sys.argv[6]))

    print "Output jpeg pixel size (resolution) = %s" % (jpeg_pixel_size)

