"""

Identify, Retrieve and  Export  Near-Cloud-Free Satellite Images over an area-of-interest
size-limit applies

Fei Zhang @Geoscience Australia

Sept 2016

"""

import os, sys

import matplotlib.pyplot as plt

from scipy import stats

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, '/g/data/u46/fxz547/Githubz/agdc-v2')  # '/home/547/fxz547/myGithub/agdc-v2')  #prepend a path

import datacube
from datacube.storage import masking

# Create an API data access object
dc = datacube.Datacube(app='GetData')


def datacube_inventory():
    """
    The Datacube provides pandas.DataFrame representations of the available products and measurements:

    :return:
    """

    # Datacube Products List
    df_prodlist = dc.list_products()

    print(df_prodlist.shape)

    print(df_prodlist.head())

    #   Measurements List of the Products
    #
    # - Each of the datacube products may have multiple measurements.
    #
    # - Measurements are related to the sensors characteristics,  also known as _bands_ in the remote-sensing domain.
    #
    # - _bands_ comes from electromagnetic wave spectral ranges, generally include Red-Green-Blue (RGB).


    df_measlist = dc.list_measurements()

    print(df_measlist.shape)

    print (df_measlist.head(5))


#####################################################
def show_images(nbar, itime):
    """
    Display NBAR single band images; select a time-slice of the imagery stack
    use pandas-style slicing to select a time period
    """

    print ("Showing images acquired in datetime ", nbar.time.values[itime])

    red_img = nbar.red.isel(time=itime)
    green_img = nbar.green.isel(time=itime)
    blue_img = nbar.blue.isel(time=itime)

    plt.figure(figsize=(16, 14))

    plt.subplot(1, 3, 1)
    plt.imshow(red_img)  # , cmap='gray')
    plt.title("Red band");
    plt.xlabel('easting');
    plt.ylabel('northing')
    plt.colorbar(orientation='vertical', shrink=0.3, label='red sensor measurement');

    plt.subplot(1, 3, 2)
    plt.imshow(green_img)  # plotting the subset data directly
    # plt.imshow(green_img, cmap='gray')
    plt.title("Green band");
    plt.xlabel('easting');
    plt.ylabel('northing')
    plt.colorbar(orientation='vertical', shrink=0.3, label='green sensor measurement');

    plt.subplot(1, 3, 3)
    plt.imshow(blue_img)  # plotting the subset data directly
    # plt.imshow(blue_img, cmap='gray')
    plt.title("Blue band");
    plt.xlabel('easting');
    plt.ylabel('northing')
    plt.colorbar(orientation='vertical', shrink=0.3, label='blue sensor measurement');

    return


# define a scale function to stretch an image

def scale_array(arr, prcnt, min_val, max_val, nan_val):
    # """
    # Linearly scales array 'arr' at the 'prcnt' percentile between 'min_val' and 'max_val',
    # replacing 'nan_val' values with NaN's.
    # #f_arr = 1.0*arr #.astype('float')    # required for NaN's
    # """

    f_arr = arr.astype('float')
    # f_arr[f_arr==nan_val] = np.nan
    prcnt_delta = (100 - prcnt) / 2
    clip_lim = np.nanpercentile(f_arr, (prcnt_delta, 100 - prcnt_delta))
    f_arr = np.clip(f_arr, clip_lim[0], clip_lim[1])
    f_arr = (f_arr - clip_lim[0]) / (clip_lim[1] - clip_lim[0])
    f_arr = f_arr * (max_val - min_val) + min_val

    return f_arr


def scale_array2(arr, prcnt, min_val, max_val, nan_val):
    """
    Linearly scales array 'arr' fake_saturation value.
    If the surf reflectance value is too large, it will be set as fake_saturation.
    What happen to the -999 values?

    return scaled  2D array image with value in (0,255)
    """

    fake_saturation = 3000
    # f_arr = arr.astype('float')
    # f_arr[f_arr==nan_val] = np.nan

    clipped_arr = arr.where(arr < fake_saturation).fillna(fake_saturation)

    return (255 * clipped_arr) / fake_saturation


def make_rgb_images(nbar, itime, outfname=None):
    """
    Create a RGB image using bands acquired at itime
    """
    print ("RGB image acquired in datetime ", nbar.time.values[itime])

    plt.figure(figsize=(10, 10))

    red_img = nbar.red.isel(time=itime)
    green_img = nbar.green.isel(time=itime)
    blue_img = nbar.blue.isel(time=itime)
    y_size = red_img.shape[0];
    x_size = red_img.shape[1]

    print (y_size, x_size)
    # print red_img.shape

    nodatav=nbar.red.nodata
    percent =99.99
    maxv=255 # 32767
    sB1data = scale_array2(red_img, percent, 0, maxv, nodatav)
    sB2data = scale_array2(green_img, percent, 0, maxv, nodatav)
    sB3data = scale_array2(blue_img, percent, 0, maxv, nodatav)

    rgb_image = np.zeros((y_size, x_size, 3), dtype='uint8')
    rgb_image[:, :, 0] = sB1data;
    rgb_image[:, :, 1] = sB2data;
    rgb_image[:, :, 2] = sB3data

    strDate = str(nbar.time.values[itime])[:10]
    title_str = 'Landsat Image over the Area: %s, %s, %s' % (str(xp), str(yp), strDate)
    plt.title(title_str)
    plt.ylabel('Northing');
    plt.xlabel('Easting');

    if outfname is None:
        plt.imshow(rgb_image)
        output_figure_name = 'nbar_nature_color.png'
        plt.savefig(output_figure_name, dpi=400)
    else:
        plt.imsave(outfname, rgb_image)  # only the image would NOT save the title and axis label
        plt.close()


    return


# # Export to geotiff raster file

GEOTIFF_DEFAULT_PROFILE = {
    'blockxsize': 256,
    'blockysize': 256,
    'compress': 'lzw',
    'driver': 'GTiff',
    'interleave': 'band',
    'nodata': 0.0,  # -999
    'photometric': 'RGB',
    'tiled': True}


def write_geotiff(filename, dataset, time_index=None, profile_override=None):
    """
    Write an xarray dataset to a geotiff

    :attr bands: ordered list of dataset names
    :attr time_index: time index to write to file
    :attr dataset: xarray dataset containing multiple bands to write to file
    :attr profile_override: option dict, overrides rasterio file creation options.
    """
    profile_override = profile_override or {}

    dtypes = {val.dtype for val in dataset.data_vars.values()}
    assert len(dtypes) == 1  # Check for multiple dtypes

    profile = GEOTIFF_DEFAULT_PROFILE.copy()
    profile.update({
        'width': dataset.dims[dataset.crs.dimensions[1]],
        'height': dataset.dims[dataset.crs.dimensions[0]],
        'affine': dataset.affine,
        'crs': dataset.crs.crs_str,
        'count': len(dataset.data_vars),
        'dtype': str(dtypes.pop()),
        'nodata': dataset.red.nodata
    })

    profile.update(profile_override)

    with rasterio.open(filename, 'w', **profile) as dest:
        for bandnum, data in enumerate(dataset.data_vars.values(), start=1):
            dest.write(data.isel(time=time_index).data, bandnum)


def export_datasets(ds, fn_prefix):
    """
    export a dataset to files with fname_prefix
    :param ds: Dataset
    :param fn_prefix: output filename prefix
    :return:
    """

    for itime in xrange(0, len(ds.time)):
        fntime = str(ds.time.values[itime])[:19].replace(':', '')
        fname = "%s_%s.tiff" % (fn_prefix, fntime)
        fname2 = "%s_%s.png" % (fn_prefix, fntime)

        path2fname = os.path.join("/short/v10/fxz547/Dexport", fname)
        path2fname2 = os.path.join("/short/v10/fxz547/Dexport", fname2)

        write_geotiff(path2fname, ds, itime)

        make_rgb_images(ds, itime, path2fname2)


def filter_export_datasets(ds, fn_prefix, timesfilter):
    """ Export a dataset time-series stack into geotiff and png files
    Filter the dataset by timesfilter.
    :param ds:
    :param fn_prefix:
    :param timesfilter: a list of timestamp
    :return:
    """
    for itime in xrange(0, len(ds.time)):
        # print (ds.time.values[itime])
        if ds.time.values[itime] in timesfilter:
            print (itime)

            fntime = str(ds.time.values[itime])[:19].replace(':', '')
            fname = "%s_%s.tiff" % (fn_prefix, fntime)
            fname2 = "%s_%s.png" % (fn_prefix, fntime)

            path2fname = os.path.join("/short/v10/fxz547/Dexport", fname)
            path2fname2 = os.path.join("/short/v10/fxz547/Dexport", fname2)

            write_geotiff(path2fname, ds, itime)

            make_rgb_images(ds, itime, path2fname2)


##########################################################################################################
# ## Group  by solar day function
#
# - There is a bit of overlap between two adjacent scenes ("cut").
# - To remove the overlap duplication, we combine the data slices with datetimes less than a minute apart.
# - Now we have fewer timeslices than found previously without solar-day-grouping
# - Landsat cycle = 16 days, 365/16 = 22 re-visits per place per year
#   For some region like Canberra, there will be overlap between passes. It will be more than 22 revisits.
#   see the Clear Observation Layer at: http://eos-test.ga.gov.au/geoserver/www/remote_scripts/WOfS_v1.6.htm

# ## Clean and near-cloudless images in AGDC
#

"""


# In[16]:

# inspect the xarray.Datasets

bands

print(bands.time.min())
print(bands.time.max())
print (bands.crs)
print(bands.data_vars)
print (bands.coords)

# nbar vs pq
print (bands.dims)
print (pq.dims)

print(bands.geobox);
print(pq.geobox)

print (bands.indexes)
print (pq.indexes)

# In[17]:

print ('No data values', bands.red.nodata)



# ## Statistics Median and Mean Images
#
# ### Normalised Vegetation Index vs Water Index

# In[ ]:

plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
mostly_good_ndvi.median(dim='time').plot()
plt.title("Median Normalised Difference Vegetation Index - NDVI");
plt.xlabel('easting');
plt.ylabel('northing')

plt.subplot(2, 2, 2)
mostly_good_ndvi.mean(dim='time').plot()
# ndvi.mean(dim='time').plot()
plt.title("Mean Normalised Difference Vegetation Index - NDVI");
plt.xlabel('easting');
plt.ylabel('northing')

# ------------------------------
plt.subplot(2, 2, 3)
mostly_good_ndwi.median(dim='time').plot()
plt.title("Median Normalised Difference Water Index - NDWI");
plt.xlabel('easting');
plt.ylabel('northing')

plt.subplot(2, 2, 4)
mostly_good_ndwi.mean(dim='time').plot()
# ndwi.mean(dim='time').plot()
plt.title("Mean Normalised Difference Water Index - NDWI");
plt.xlabel('easting');
plt.ylabel('northing')

# # Water Observation from Space
#
# http://eos-test.ga.gov.au/geoserver/www/remote_scripts/WOfS_v1.6.htm
#

# ## DSM dataset indexed in DC

# ## Re-projection and Re-sampling

# http://spatialreference.org/ref/epsg/gda94-geoscience-australia-lambert/ 3112
# EPSG:3577: GDA94 / Australian Albers
# EPSG:3112 GDA94 / Geoscience Australia Lambert
# Canberra   grid = dc.load(product='dsm1sv10', x=(149.07, 149.17), y=(-35.25, -35.35))
grid = dc.load(product='dsm1sv10', x=xp, y=yp)
grid.elevation
grid.elevation[0].plot()

query = {
    'lat': (-35.2, -35.4),
    'lon': (149.0, 149.2),
}

dsmgrid = dc.load(product='dsm1sv10', output_crs='EPSG:4326', resolution=(-0.00025, 0.00025), **query)

"""

####################################################################################################################

if __name__ == "__main__":

    datacube_inventory()

    # Select best RGB NDVI NDWI images: eg,70% good valid pixels
    Good_Pixels_Pct = 90 

    # Defina an Area of Interest

    # Where?
    # AOI_NAME = 'LakeBG';xp = (149.07, 149.17); yp = (-35.25, -35.35)  #Lake Burley Griffin
    # xp=( 149.160,  149.170); yp=(-35.34 , -35.35 )  #a small area cover GA water pond

    # Edge of fire scar (NT) 19:36:10.24S//132:34:55.72E
    # 0.1degree =~ 10km
    #AOI_NAME = 'FireScarNT';  xp = (132.50, 132.60);    yp = (-19.65, -19.55)

    # Farming land (crops (green and fallow paddocks - Western NSW)30:5:45.33S/148:11:29.75E
    #AOI_NAME='Farmland'; xp=(148.14, 148.24); yp=(-30.15, -30.05)

    # Woodland (greenish most of the time. the understory should change in response to seasonal drying out)22:52:35.4S/147:24:11.25E
    #AOI_NAME='Woodland'; xp=(147.35, 147.45); yp=(-22.92, -22.82)

    # #GungahlinACT
    # AOI_NAME = 'CanberraNorth'; xp= (149.061024, 149.163758); yp=(-35.217105, -35.151768)
    #
    AOI_NAME = 'LakeGeorge';
    xp = (149.28, 149.53);
    yp = (-34.98, -35.25)  # Lake George area

    # When? a time range
    tp = ('2001-01', '2016-12-31')

    # what product type?
    prod_type = 'ls8_nbar_albers'
    # associated pq product code
    pq_prod_type = prod_type.replace('nbar', 'pq')

    # output grid and resolution?
    v_output_crs = None;
    v_resolution = None  # None if AGDC default

    # v_output_crs='EPSG:3112'; v_resolution=(-25,25)  #LCC 3112, Albers 3577
    # output_crs='EPSG:4326', resolution=(-0.00025, 0.00025)


    #  retrieve a subset of bands
    # bands = dc.load(product=prod_type, x=xp, y=yp, time=tp, group_by='solar_day',
    #        measurements=['red', 'nir', 'green', 'swir1','blue'])

    # retrieve all bands
    bands = dc.load(product=prod_type, x=xp, y=yp, time=tp, group_by='solar_day')

    red = bands.red.where(bands.red != bands.red.attrs['nodata'])
    nir = bands.nir.where(bands.nir != bands.nir.attrs['nodata'])

    green = bands.green.where(bands.green != bands.green.attrs['nodata'])
    swir1 = bands.swir1.where(bands.swir1 != bands.swir1.attrs['nodata'])

    # Retrieve the associated Pixel Quality datasets. pq has issue with group_by_solar_day
    # They may not match all the Nbar products

    pq = dc.load(product=pq_prod_type, x=xp, y=yp, time=tp, group_by='solar_day')  # , fuse_func='')

    cloud_free = masking.make_mask(pq, cloud_acca='no_cloud', cloud_fmask='no_cloud', contiguous=True).pixelquality

    # The returned data is an `xarray.Dataset` object, which is a labelled n-dimensional array wrapping a `numpy` array.
    #
    # We can investigate the data to see the variables (measurement bands) and dimensions that were returned:
    #
    # We can look at the data by name directly, or through the `data_vars` dictionary:
    #


    # Compute Indexes: NDVI and NDWI: bands math, Numpy array arithmetics without looping

    # Normalized Differenc Vegetation Index: Red and near Infrared bands

    ndvi = ((nir - red) / (nir + red)).where(cloud_free)

    # Normalized Difference Water Index: Green and Shortwave Infrared Bands

    ndwi = ((swir1 - green) / (green + swir1)).where(cloud_free)
    # ndwi = ((green- swir1)/(green + swir1)).where(cloud_free)


    # Weed out the low quality images.

    # xarray.Dataset.sum to reduce the datasets by selecting the time slices with high percentage cloud-free pixels

    cloudfreeRatio = 0.01 * Good_Pixels_Pct  # threshold of cloud pixel 70% ??

    mostly_cloud_free = cloud_free.sum(dim=('x', 'y')) > (cloudfreeRatio * cloud_free.size / cloud_free.time.size)

    print(mostly_cloud_free)

    # How many images selected?

    print(mostly_cloud_free.sum().values)

    good_times = []
    for itime in xrange(0, len(mostly_cloud_free)):
        if mostly_cloud_free[itime].values:
            print (itime, mostly_cloud_free[itime].values, mostly_cloud_free[itime].time.values)
            good_times.append(mostly_cloud_free[itime].time.values)

    print(good_times)

    # NDVI: Apply the time-dim mask to the 3D-array (time, x, y)
    # apply the cloud_threshold mask, which will select a subset images with good pixels.

    # mostly_good_ndvi = ndvi.where(mostly_cloud_free).dropna('time', how='all')
    # mostly_good_ndvi.plot(col='time', col_wrap=5)


    mostly_good_ndwi = ndwi.where(mostly_cloud_free).dropna('time', how='all')

    # NOT work in standalone py
    # mostly_good_ndwi.plot(col='time', col_wrap=5)

    # filter_export_datasets(bands, "CleanImages_LakeGorge", good_times)
    fnpref = '%s_%s' % (AOI_NAME, prod_type[:3])
    filter_export_datasets(bands, fnpref, good_times)

    # ffmpeg -r 1 -pattern_type glob -i 'Dexport/CleanImages_LakeGorge_*.png' -c:v libx264 LS78.mp4

    # Display the nearly cloud free images [0, 8, 9, 11, 12, 20, 24, 26, 30, 32, 34]
    plt.figure(figsize=(12, 12))
    it = 8
    make_rgb_images(bands, it)
    print ("where is the figure?")
