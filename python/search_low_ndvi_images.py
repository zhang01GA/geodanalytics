"""
Description:
    find low NDVI images for bare soil observation (Llandsat 5,7,8)
    For an given spatial-temporal area of interest bounding box,
    retrieve images calculate NDVI, filter and mean the NDVI images,
    in order to find the low NDVI images for further processing.
    The ultimate purpose is to develop high-quality datasets for the mineral uncover machine-learning application.

Inputs: configuration file

Outputs: a list of datetimes, each of which identifies a low NDVI image.

Fei Zhang @Geoscience Australia

Sept 2016

Below are the centre point locations for 3 study areas with different vegetation/bareness relationships;

1. Edge of fire scar (NT) 19:36:10.24S//132:34:55.72E

2. Farming land (crops (green and fallow paddocks - Western NSW)30:5:45.33S/148:11:29.75E

3. Woodland (remain greenish most of the time - the understory should change in response to seasonal drying out)22:52:35.4S/147:24:11.25E

extract a 5X5 km or 10X10km tile for each area. Then select only tiles with < 10% poor quality pixels (e.g. noise , cloud etc.).
From that subset we run NDVI and then run a median kernel (3*3 ?) to smooth out noisy pixels. The kernel needs to accommodate crappy pixels as no data values. After that we calculate the average NDVI response for whole the tile. We should try a keep the tiles in chronological order because when we process all the tiles through time series and plot up the average NDVI value for each tile we will want to see trends of drying out (increased bareness) due to seasonal effects or recovery from fire scars. I suggest we look at the image tiles along each step of the work flow. There are lots of parameters (size of tile, size and shape of the kernel etc) we can change  - but as a first step this might be a good start.

"""

import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

import pandas as pd
import rasterio

sys.path.insert(0, '/g/data/u46/fxz547/Githubz/agdc-v2')  # '/home/547/fxz547/myGithub/agdc-v2')  #prepend a path

import datacube
from datacube.storage import masking

import shutil
from datetime import datetime
import yaml
import logging

logging.basicConfig()
_logger = logging.getLogger(__file__)  # (__name__) ()is root
_logger.setLevel(logging.INFO)
_logger.setLevel(logging.DEBUG)


class NDVI_Image_Stack:
    def __init__(self, confile=None):

        # Create an API data access

        self.dc = datacube.Datacube(app='GetData')

        if confile is None:
            # default values:

            # When? a time range
            self.tp = ('2015-01-01', '2016-12-31')

            # Where?

            # xp=(149.00, 149.20); yp=(-35.25, -35.35)  # Lake Burley Griffin x=(149.07, 149.17), y=(-35.25, -35.35),
            # xp=( 149.160,  149.170); yp=(-35.34 , -35.35 )  #a small area cover GA water pond


            # Edge of fire scar (NT) 19:36:10.24S//132:34:55.72E
            # 0.1degree =~ 10km
            # self.AOI_NAME='FireScarNT'; self.xp=(132.50, 132.60); self.yp=(-19.65, -19.55)

            # Farming land (crops (green and fallow paddocks - Western NSW)30:5:45.33S/148:11:29.75E
            # self.AOI_NAME='Farmland'; self.xp=(148.14, 148.24); self.yp=(-30.15, -30.05) # North of Dubbo, inside NSW border

            # Woodland (greenish most of the time. the understory should change in response to seasonal drying out)22:52:35.4S/147:24:11.25E
            self.AOI_NAME = 'Woodland';
            self.xp = (147.35, 147.45);
            self.yp = (-22.92, -22.82)  # west of Rockhampton
            # woodland has a blank image in 2016-05? with NDVI=0. how to exclude it?


            self.prod_type = 'ls8_nbar_albers'

            # self.pq_prod_type = self.prod_type.replace('nbar', 'pq')

            self.qdict ={'AOI_NAME':self.AOI_NAME, "lon": self.xp, "lat": self.yp, "time": self.tp, 'prod_type': self.prod_type}

            _logger.debug(self.qdict)

        else:
            self.qdict = self.init_from_config(confile)

    def init_from_config(self, configfile):
        """
        Initialize the object self attributes from a config file
        :param conf_file: path2afile
        :return:
        """
        _logger.debug("parsing configuration file %s to get input parameters"% configfile)

        with open(configfile, 'r') as f:
            config = yaml.safe_load(f)

        self.prod_type = config.get('prod_type')

        self.AOI_NAME = config.get('aoi_name')

        # determine spatial coverage
        self.yp = (
            float(config.get('lat_min_deg')), float(config.get('lat_max_deg')))

        self.xp = (
            float(config.get('lon_min_deg')), float(config.get('lon_max_deg')))

        # determine time period

        self.tp = (config.get('start_datetime'), config.get('end_datetime'))
        # dateutil.parser.parse(self.config.get('coverage','start_datetime')), \
        # dateutil.parser.parse(self.config.get('coverage','end_datetime'))  )

        self.qdict = {'AOI_NAME':self.AOI_NAME, "lon": self.xp, "lat": self.yp, "time": self.tp, 'prod_type': self.prod_type}

        _logger.debug(self.qdict)

        return self.qdict

    def update_from_dict(self, new_qdict):
        """
        update the original query dict by a new_qdict
        :param new_qdict:
        :return:
        """
        self.qdict.update(new_qdict)

        self.AOI_NAME = self.qdict.get('AOI_NAME')
        self.xp = self.qdict.get('lon')
        self.yp = self.qdict.get('lat')
        self.tp = self.qdict.get('time')

        self.prod_type= self.qdict.get('prod_type')

        _logger.debug(self.qdict)

        return self.qdict



    def get_valid_data(self, prod_type):

        query = {
            'time': ('1990-01-01', '1991-01-01'),
            'lat': (-35.2, -35.4),
            'lon': (149.0, 149.2),
        }

        mslist = ['red', 'nir']  # ,'green', 'blue']
        data = dc.load(product=prod_type, measurements=mslist, group_by='solar_day', **query)
        data = masking.mask_valid_data(data)

        return data

    def get_ndvi(self, prod_type, cloudfreeRatio=0.5):
        """
        get NDVI images for the prod_type, select cloud free ratio better.
        """

        blist = ['red', 'nir']  # , 'green', 'swir1']
        bands = self.dc.load(product=prod_type, x=self.xp, y=self.yp, time=self.tp, measurements=blist,
                             group_by='solar_day')

        # mask off nodata pixels
        # red = bands.red.where(bands.red != bands.red.attrs['nodata'])
        # nir = bands.nir.where(bands.nir != bands.nir.attrs['nodata'])
        # green = bands.green.where(bands.green != bands.green.attrs['nodata'])
        # swir1 = bands.swir1.where(bands.swir1 != bands.swir1.attrs['nodata'])
        # Or equivalently
        #     red= masking.mask_valid_data(bands.red)
        #     nir= masking.mask_valid_data(bands.nir)


        bands = masking.mask_valid_data(bands)
        red = bands.red
        nir = bands.nir

        # Retrieve the associated Pixel Quality datasets ls8_pq_albers
        pq_prod_type = prod_type.replace('nbar', 'pq')
        pq = self.dc.load(product=pq_prod_type, x=self.xp, y=self.yp, time=self.tp, group_by='solar_day')

        print ("NBAR and PQ slices: %s %s" % (len(bands.time), len(pq.time)))

        # to get perfect good pixels, how about cloud shadows?
        cloud_free = masking.make_mask(pq, cloud_acca='no_cloud', cloud_fmask='no_cloud',
                                       cloud_shadow_acca='no_cloud_shadow', cloud_shadow_fmask='no_cloud_shadow',
                                       contiguous=True).pixelquality
        # cloud_free = masking.make_mask(pq, cloud_acca='no_cloud', cloud_fmask='no_cloud', contiguous=True).pixelquality
        print(cloud_free)

        ndvi = ((nir - red) / (nir + red)).where(cloud_free)  # mask off the cloud etc (again after mask_valid_data()?)

        # red1 = red.where(cloud_free)
        # nir1 = nir.where(cloud_free)
        # ndvi = ((nir1 - red1) / (nir1 + red1))  # .where(cloud_free)  ?

        # Normalized Differenc Water Index: Green and Shortwave Infrared Bands

        # ndwi = ((swir1- green)/(green + swir1)).where(cloud_free)

        # ndwi = ((green- swir1)/(green + swir1)).where(cloud_free)

        print("NDVI shape ", str(ndvi.shape))

        # xarray.Dataset.sum to reduce the datasets by selecting the time slices with high percentage cloud-free pixels
        #  Weed out the low quality images
        # cloudfreeRatio = 0.8  # threshold of cloud pixel 80%

        mostly_cloud_free = cloud_free.sum(dim=('x', 'y')) > (cloudfreeRatio * cloud_free.size / cloud_free.time.size)

        print(mostly_cloud_free)

        print("How many cloudless images selected?", mostly_cloud_free.sum().values)

        # Apply the time-dim mask to the 3D-array (time, x, y)
        mostly_good_ndvi = ndvi.where(mostly_cloud_free).dropna('time', how='all')
        # mostly_good_ndvi.plot(col='time', col_wrap=5)

        print(mostly_good_ndvi)

        return mostly_good_ndvi

    def get_ndvi_mean(self, ndvimg):
        """
        nanmean?
        """

        ndvi_mean = ndvimg.mean(dim=['x', 'y'])  # average over the image pixels

        pdser = ndvi_mean.to_pandas()  # pd.Series

        df = pdser.to_frame(name='NDVI')  # convert to dataframe

        # df['PROD_TYPE'] = self.prod_type  # add a new column

        return df.sort_index()

    @staticmethod
    def filter_center(A, size=3, no_data_val=None, func=np.nanmean):
        """
        nanmean of the matrix A filtered by a uniform kernel of size=size
        This function `centers` the kernel at the target pixel.
        This is slightly different from scipy.ndimage.uniform_filter application.
        In scipy.ndimage.uniform_filter, a convolution approach is implemented.
        An equivalent is scipy.ndimage.uniform_filter like convolution approach with
        no_data_val/nan handling can be found in filter_broadcast_uniform_filter in
        this module.
        ----------
        A = input dataarray
        size = odd number uniform filtering kernel size
        no_data_val = value in matrix that is treated as no data value
        func: function to use, choose from np.nanmean nanmedian, nanmax, nanmin as required.
        Returns: a filtered image matrix of the same shape as input A

        ----------
        Ref:
        https://github.com/NICTA/uncover-ml/blob/develop/preprocessing/raster_average.py
        http://stackoverflow.com/questions/23829097/python-numpy-fastest-method-for-2d-kernel-rank-filtering-on-masked-arrays-and-o?rq=1
        Notes
        """

        from numpy.lib.stride_tricks import as_strided

        assert size % 2 == 1, 'Please supply an odd size'
        rows, cols = A.shape

        padded_A = np.empty(shape=(rows + size - 1,
                                   cols + size - 1),
                            dtype=A.dtype)
        padded_A[:] = np.nan
        rows_pad, cols_pad = padded_A.shape

        if no_data_val:
            mask = A == no_data_val
            A[mask] = np.nan

        padded_A[size // 2:rows_pad - size // 2, size // 2: cols_pad - size // 2] = A.copy()

        N, M = A.shape

        B = as_strided(padded_A, (N, M, size, size),
                       padded_A.strides + padded_A.strides)
        B = B.copy().reshape((N, M, size ** 2))

        return func(B, axis=2)

    def pipeline(self, prod):
        """
        For the given prod_type, compute average NDVI time series, both original NDVI and spatially-filtered
        return a merged pandas df
        """

        if prod is None:
            prodtype = self.prod_type
            print("Using self prod_type", prodtype)
        else:
            prodtype=prod

        # Compute the original NDVI stack timeseries (nir-red/nir+red)

        ndvi_stack = self.get_ndvi(prodtype, cloudfreeRatio=0.5)

        print (ndvi_stack.shape)

        # imshow a few of the ndvi images
        # ndvi_stack[:20].plot(col='time', col_wrap=5, add_colorbar=False)

        spdf = self.get_ndvi_mean(ndvi_stack)  # original ndvi nanmean time-series, sorted

        # loop over filters on NDVIs
        max_filtersize = 5  #filter sizes= [3, 5, ...2*maxfs +1]
        new_cols = ['NDVI0']

        for fs in xrange(1, max_filtersize+1):
            fsize=2*fs+1

            _logger.debug('Doing filter of size= %s'%str(fsize))

            new_cols.append('NDVI%s'%(str(fsize)))
            new_spdf= filtered_ndvi_stats(ndvi_stack, ndisk=fsize)
            spdf = pd.merge(spdf, new_spdf, left_index=True, right_index=True, how='outer')


        spdf.columns = new_cols

        spdf['PROD_TYPE'] = prodtype  # add a new column

        return spdf

        # pdf5 = filtered_ndvi_stats(ndvi_stack, ndisk=5)
        # spdf5 = pdf5.sort_index()
        #
        # pdf10 = filtered_ndvi_stats(ndvi_stack, ndisk=11)
        # spdf11 = pdf10.sort_index()
        # # pdf10.plot(figsize=(20, 10), marker='*')
        #
        # # fsize = 21
        # pdf21 = filtered_ndvi_stats(ndvi_stack, ndisk=21)
        # spdf21 = pdf21.sort_index()
        #
        # for irow in xrange(0, spdf.shape[0]):
        #     dtime = spdf.index[irow]
        #     print(dtime, spdf.iloc[irow][0], spdf5.iloc[irow].NDVI, spdf11.iloc[irow].NDVI, spdf21.iloc[irow].NDVI)
        #
        # ndvidf5 = pd.merge(spdf, spdf5, left_index=True, right_index=True, how='outer')
        #
        # ndvidf11 = pd.merge(ndvidf5, spdf11, left_index=True, right_index=True, how='outer')
        #
        # ndvidf21 = pd.merge(ndvidf11, spdf21, left_index=True, right_index=True, how='outer')
        #
        # ndvidf21.head()
        #
        # # Rename columns
        # new_cols = ['NDVI0', 'NDVIS5', 'NDVIS11', 'NDVIS21']
        # ndvidf21.columns = new_cols
        #
        # ndvidf21['PROD_TYPE'] = prod  # add a new column
        #
        # return ndvidf21

    def main(self, prod=None):
        """
        main function to test run the pipeline: get a merged pdfame of NDVI time series, plot and to_csv the data.
        :return:
        """
        if prod is None:
            prod=self.prod_type

        pdframe = self.pipeline(prod)

        # pdframe.plot(figsize=(20, 10), marker='o')


        outcsvfile = '%s_%s_NDVI.csv' % (self.AOI_NAME, prod)
        pdframe.to_csv(outcsvfile)  # ("Woodland_LS5_NDVI.csv")

        return pdframe


######################################################################

def test_filter(img1):
    """
    test a particular image over the filters function
    :param img1:
    :return:
    """

    # img1 = good_ndvi[7]  # pick one image from the stack

    img1f = NDVI_Image_Stack.filter_center(img1, size=3, func=np.nanmedian)

    plt.imshow(img1);
    plt.imshow(img1f)

    img1.mean()
    np.nanmean(img1f)

    df_ndvi_ls8 = get_ndvi_mean(good_ndvi)
    df_ndvi_ls8.sort_values('NDVI').head(10)


def multi_sensor_ndvi():
    """
    To do clean up
    :return:
    """

    # prod_type='ls7_nbar_albers'
    # df_ndvi_ls7=get_ndvi_mean(good_ndvi)

    # # ls5
    # prod_type='ls5_nbar_albers'
    # df_ndvi_ls5=get_ndvi_mean(good_ndvi)

    # ndvi578=pandas.concat([df_ndvi_ls5, df_ndvi_ls7, df_ndvi_ls8])

    # if only Landsat-8

    ndvi578 = df_ndvi_ls8

    ndvi578.sort_values('NDVI').head(40)  # sort ndvi values

    ndvi578.shape

    ndvi578.plot(figsize=(20, 10), marker='*')

    outcsvfile = 'meanNDVI578_%s.csv' % (AOI_NAME)

    ndvi578.to_csv(outcsvfile)  # ('/tmp/meanNDVI578_FireScarNT.csv')

    ndvi578.hist(bins=100)

    p10 = ndvi578.quantile(0.1)
    p90 = ndvi578.quantile(0.9)

    bot_tenperc = ndvi578[(ndvi578['NDVI'] <= p10[0])].dropna()
    top_tenperc = ndvi578[(ndvi578['NDVI'] >= p90[0])].dropna()

    outcsvfile2 = 'meanNDVI578_%s_bot10pc.csv' % (AOI_NAME)
    bot_tenperc.to_csv(outcsvfile2)

    outcsvfile3 = 'meanNDVI578_%s_top10pc.csv' % (AOI_NAME)
    top_tenperc.to_csv(outcsvfile3)

    top_tenperc.head

    print(p10)

    bot_tenperc.head(100)

    return


def filtered_ndvi_stats(ndvi_imgs, ndisk=3, func=np.nanmedian):
    """
    New spatial filter to a stack of input ndvi_imgs, which may have nan-valued pixels,
    ndisk=3 is the default size of the  filter
    return a pandas dataframe of mean NDVI for the images.

    """

    mydict = {}
    for it in xrange(0, len(ndvi_imgs.time)):
        # apply median filter to get an image meds for this timeslice
        img = ndvi_imgs.isel(time=it)
        acqdate = ndvi_imgs.time[it].values

        filtered_img = NDVI_Image_Stack.filter_center(img, size=ndisk, func=func)

        # show this image?  plt.imshow(filtered_img)

        mydict.update({acqdate: np.nanmean(filtered_img)})  # mean ndvi of the image

    # convert mydict to pandas dataframe, with proper column names and index
    pdf = pd.DataFrame(mydict.items(), columns=['Date', 'NDVI'])
    pdf.set_index('Date', inplace=True)

    return pdf.sort_index()


# notworking @deprecated
def old_filtered_ndvi_mean(ndvi_imgs, ndisk=5):
    """
    Apply a spatial filter to the input ndvi_imgs array
    return a pandas dataframe of mean NDVI for the images.
    ndisk=5 is the default size of the disk filter
    See http://scikit-image.org/docs/stable/api/skimage.filters.html?highlight=local%20median%20filter
    
    scaled_good_ndvi = 128 * (1 + good_ndvi)  # to unint8 or uint16 for scikit-image filter input

    uint8_good_ndvi = scaled_good_ndvi.astype('uint8')

    pdf5 = filtered_ndvi_mean(uint8_good_ndvi, ndisk=5)
    """

    # http://scikit-image.org/docs/stable/api/skimage.filters.html?highlight=local%20median%20filter

    from skimage import data
    from skimage.morphology import disk
    from skimage.filters.rank import median

    mydict = {}
    for it in xrange(0, len(ndvi_imgs.time)):
        # apply median filter to get an image meds for this timeslice
        img = ndvi_imgs.isel(time=it)
        imask = ~np.isnan(img)
        meds = median(img, disk(ndisk), mask=imask)
        # plt.imshow(meds)
        mydict.update({ndvi_imgs.time[it].values: meds.mean()})

    # convert mydict to pandas dataframe, with proper column names and index
    pdf = pd.DataFrame(mydict.items(), columns=['Date', 'NDVI'])
    pdf.set_index('Date', inplace=True)

    return pdf


############################################################################################
# ## Statistics Median and Mean Images
#
# ### Normalised Vegetation Index vs Water Index


# plt.figure(figsize=(16, 12))
#
# plt.subplot(2, 2, 1)
# mostly_good_ndvi.median(dim='time').plot()
# plt.title("Median Normalised Difference Vegetation Index - NDVI");
# plt.xlabel('easting');
# plt.ylabel('northing')
#
# plt.subplot(2, 2, 2)
# mostly_good_ndvi.mean(dim='time').plot()
# # ndvi.mean(dim='time').plot()
# plt.title("Mean Normalised Difference Vegetation Index - NDVI");
# plt.xlabel('easting');
# plt.ylabel('northing')
#
# # ------------------------------
# plt.subplot(2, 2, 3)
# mostly_good_ndwi.median(dim='time').plot()
# plt.title("Median Normalised Difference Water Index - NDWI");
# plt.xlabel('easting');
# plt.ylabel('northing')
#
# plt.subplot(2, 2, 4)
# mostly_good_ndwi.mean(dim='time').plot()
# # ndwi.mean(dim='time').plot()
# plt.title("Mean Normalised Difference Water Index - NDWI");
# plt.xlabel('easting');
# plt.ylabel('northing')

########################################################
# Thanks John for the summary.
# That's what I understand mostly.
#
# "	I have changed the filter to use Sudipta's function which handle nan correctly. But it's slow.
# Because the scikit-image function did not hand nan correctly and give mislead results
# "	I have make cloud ratio a parameter. Can be 50% or any. The cloud pixels will be masked out (excluded) for NDVI calculation.
# "	I will identify the low NDVI images from the stacks, and produce a list of acq datetime, so that Dale's process can pick up as inputs.
# "	The results of Dale's processing will be synthetic images representing low NDVI observations.
# "	Then John will evaluate the synthtic images.
#
# Cheers
#
# Fei
#
#
# From: Wilford John
# Sent: Friday, 30 September 2016 11:30 AM
# To: Zhang Fei; Roberts Dale
# Subject: bare earth [SEC=UNCLASSIFIED]
#
# Hi Fei and Dale,
#
# From the meeting yesterday this is what I understand we are going to do.
#
# 1. Include both Landsat 5 and 8 in the analysis
#
# 2. We talked about relaxing the cloud threshold to 50% - but I think we are going to get enough scenes by including a deeper temporal selection with the current threshold?
#
# 3. We re-run the median NDVI filters on the new temporal stack
#
# 4. Plot up NDVI trends (average NDVI for each image). On the same graph we plot the diffent kernel sizes averages.
#
# 5. From the plot we select a subset of images  - i.e. the low NDVI range*. - can you work together on this because I'm away next week.
#
# 6. These images are sent to Dale for processing
#
# 7. After processing I will run tests and compare the new dataset with a single time slice image.
#
# Fei you are correct in that we are not dealing with the removal of dry vegetation - lets first try this simple approach and build from there. I have a feeling we will need to reduce the size of the images to be more sensitive to local  greenness changes.
#
# *my suggestion is that we avoid extremely low NDVI because they might be noise (poor cloud removal, bad data) but I'm happy to take your advice on this.
#
# Cheers john


######################################################################
# provide a config file to initialise the class object NDVI_Image_Stack
# How2run:
#   python search_low_ndvi_images.py woodland.yml [ ls5_nbar_albers | ls8_nbar_albers | ls7_nbar_albers]
# ---------------------------------------------------------------------
if __name__ == "__main__":

    prod=None
    confile=None

    if len(sys.argv) < 2:
        print("Using default param ")
    elif len(sys.argv) >= 2:
        confile= sys.argv[1]

    if(sys.argv) == 3:  # a product_type is specified for the AOI
        prod = sys.argv[2]

    # construct an initial object
    ndviobj = NDVI_Image_Stack(confile)


    ndviobj.main(prod)
