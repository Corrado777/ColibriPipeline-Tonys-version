import os
import sep
import numpy as np
import numba as nb
from glob import glob
from astropy.io import fits
from astropy.time import Time
from copy import deepcopy
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import os
import gc
import time as timer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from matplotlib.colors import LogNorm

# Function for reading specified number of bytes
def readxbytes(fid, numbytes):
    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
    return data

# Function to read 12-bit data with Numba to speed things up
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_data(data_chunk):
    """data_chunk is a contigous 1D array of uint8 data)
    eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""
    #ensure that the data_chunk has the right length

    assert np.mod(data_chunk.shape[0],3)==0

    out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
    image1 = np.empty((2048,2048),dtype=np.uint16)
    image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out

def getSizeRCD(filenames):
    """ MJM - Get the size of the images and number of frames """
    filename_first = filenames[0]
    frames = len(filenames)

    width = 2048
    height = 2048

    # You could also get this from the RCD header by uncommenting the following code
    # with open(filename_first, 'rb') as fid:
    #     fid.seek(81,0)
    #     hpixels = readxbytes(fid, 2) # Number of horizontal pixels
    #     fid.seek(83,0)
    #     vpixels = readxbytes(fid, 2) # Number of vertical pixels

    #     fid.seek(100,0)
    #     binning = readxbytes(fid, 1)

    #     bins = int(binascii.hexlify(binning),16)
    #     hpix = int(binascii.hexlify(hpixels),16)
    #     vpix = int(binascii.hexlify(vpixels),16)
    #     width = int(hpix / bins)
    #     height = int(vpix / bins)

    return width, height, frames

# Function to split high and low gain images
def split_images(data,pix_h,pix_v,gain):
    interimg = np.reshape(data, [2*pix_v,pix_h])

    if gain == 'low':
        image = interimg[::2]
    else:
        image = interimg[1::2]

    return image
 
# Function to read RCD file data
def readRCD(filename):

    hdict = {}

    with open(filename, 'rb') as fid:

        # Go to start of file
        fid.seek(0,0)

        # Serial number of camera
        fid.seek(63,0)
        hdict['serialnum'] = readxbytes(fid, 9)

        # Timestamp
        fid.seek(152,0)
        hdict['timestamp'] = readxbytes(fid, 29).decode('utf-8')

        # Load data portion of file
        fid.seek(384,0)

        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    return table, hdict

def process_batch(filepaths):
    low_gain_batch = []
    high_gain_batch = []
    
    hnumpix = 2048
    vnumpix = 2048
    
    for file in filepaths:
        data, header = readRCD(file)
        images = nb_read_data(data)
        low_gain = split_images(images, hnumpix, vnumpix, 'low')
        high_gain = split_images(images, hnumpix, vnumpix, 'high')

        low_gain_batch.append(low_gain)
        high_gain_batch.append(high_gain)
    
    # Stack the batch
    stacked_low_gain_batch = np.median(np.array(low_gain_batch), axis=0)
    stacked_high_gain_batch = np.median(np.array(high_gain_batch), axis=0)
    
    return stacked_low_gain_batch, stacked_high_gain_batch
    

if __name__ == "__main__":
    data_dir = "/home/agirmen/Github/ColibriPipeline_SensitivityModel/Sensitivity_Test_Data/20240729/20240729_03.05.55.471"\

    # Get the filepaths of all the files in the directory

    filepaths = []
    for root, dirs, files in os.walk(data_dir):
        files.sort()
        for file in files:
            if file.endswith(".rcd"):
                filepaths.append(os.path.join(root, file))


    batch_size = 20  # Define the batch size

    # Process images in batches

    low_gain_images = []
    high_gain_images = []   

    num_files = len(filepaths[:500])
    for i in range(0, num_files, batch_size):
        print(f"processing image {i} of {num_files}")

        batch_filepaths = filepaths[i:i + batch_size]
        stacked_low_gain_batch, stacked_high_gain_batch = process_batch(batch_filepaths)
        
        low_gain_images.append(stacked_low_gain_batch)
        high_gain_images.append(stacked_high_gain_batch)

    # Combine the stacked results of each batch
    final_stacked_low_gain = np.median(np.array(low_gain_images), axis=0)
    final_stacked_high_gain = np.median(np.array(high_gain_images), axis=0)

    # Combine the stacked results of each batch using mean
    final_stacked_low_gain = np.mean(np.array(low_gain_images), axis=0)
    final_stacked_high_gain = np.mean(np.array(high_gain_images), axis=0)

    # Save the final stacked high gain image as a .fits file
    hdu = fits.PrimaryHDU(final_stacked_high_gain)
    hdul = fits.HDUList([hdu])
    hdul.writeto('final_stacked_high_gain.fits', overwrite=True)