"""
Created 2018 by Emily Pass

Update: March 1, 2022 - Rachel Brown
Update: Sept. 21, 2022 - Peter Quigley

Update: September 19, 2022 - Roman Akhmetshyn - changed output txt file contents and added a significance calculation


-initial Colibri data processing pipeline for flagging candidate
KBO occultation events
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import datetime
import gc
import sep
import numpy as np
import time as timer
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp

# Disable Warnings
import warnings
import logging
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=VisibleDeprecationWarning)

global inner_annulus
inner_annulus = 5
global outer_annulus 
global Edge_buffer
Edge_buffer=10	


#--------------------------------functions------------------------------------#

def getSizeFITS(imagePaths):
    """
    Get number of images in a data directory and image dimensions (.fits only)
    
        Parameters:
            imagePaths (list): List of all image filepaths in data directory
        
        Returns:
            width (int): Width of .fits images
            height (int): Height of .fits images
            frames (int): Number of images in a data directory
    """
    
    '''get number of images in directory'''
    
    frames = len(imagePaths)         #number of images in directory


    '''get width/height of images from first image'''
    
    first_imagePath = imagePaths[0]             #first image in directory
    first_image = fits.open(first_imagePath)    #load first image
    header = first_image[0].header              #image header
    width = header['NAXIS1']                    #X axis dimension
    height = header['NAXIS1']                   #Y axis dimension

    return width, height, frames


def importFramesFITS(imagePaths, startFrameNum, numFrames, bias):
    """
    Reads in frames from .fits files starting at a specific frame
    
        Parameters:
            imagePaths (list): List of image paths to read in
            startFrameNum (int): Starting frame number
            numFrames (int): How many frames to read in
            bias (arr): 2D array of fluxes from bias image
            
        Returns:
            imagesData (arr): Image data
            imagesTimes (arr): Header times of these images
    """

    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]

    '''get data from each file in list of files to read, subtract bias frame (add 100 first, don't go neg)'''
    for imagePath in files_to_read:

        image = fits.open(imagePath)
        
        header = image[0].header
        
        imageData = image[0].data - bias
        headerTime = header['DATE-OBS']
        
        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        imageMinute = str(headerTime).split(':')[1]
        dirMinute = imagePath.parent.name.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #on red don't need to convert between UTC and local (local is UTC)
            newLocalHour = int(imagePath.parent.name.split('_')[1].split('.')[0])
        
            if int(imageMinute) < int(dirMinute):
               # newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1         #add 1 if hour changed over during minute
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour  = newLocalHour  
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            #newTimestamp = replaced.encode('utf-8')
            headerTime = replaced
            
        image.close()

        imagesData.append(imageData)
        imagesTimes.append(headerTime)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes


def runParallel(minuteDir, MasterBiasList, ricker_kernel, exposure_time, sigma_threshold):
    star_minutes = firstOccSearch(minuteDir, MasterBiasList, ricker_kernel, exposure_time, sigma_threshold)
    gc.collect()
    
    return star_minutes


#############
# Former main
#############

def firstOccSearch(minuteDir, MasterBiasList, kernel, exposure_time, sigma_threshold,
                   base_path,obs_date,
                   telescope='TEST',RCDfiles = True, gain_high = True):
    """ 
    Formerly 'main'.
    Detect possible occultation events in selected file and archive results 
    
        Parameters:
            minuteDir (str): Filepath to current directory
            MasterBiasList (list): List of master biases and times
            kernel (arr): Ricker wavelet kernel
            exposure_time (int/float): Camera exposure time (in __)
            sigma_threshold (float): Sensitivity of the detection filter
        
        Returns:
            __ (str): Printout of processing tasks
            __ (.npy): .npy format file with star positions (if doesn't exist)
            __ (.txt): .txt format file for each occultation event with names
                       of images to be saved
            __ (int/float): time of saved occultation event images
            __ (float): flux of occulted star in saved images
    """

    print (f"{datetime.datetime.now()} Opening: {minuteDir}")

    ''' create folder for results '''
    
    day_stamp = obs_date
    savefolder = base_path.joinpath('ColibriArchive', str(day_stamp))
    if not savefolder.exists():
        savefolder.mkdir()      

    '''load in appropriate master bias image from pre-made set'''
    bias = cir.chooseBias(minuteDir, MasterBiasList, obs_date)


    ''' adjustable parameters '''
    
    ap_r = 3.   #radius of aperture for flux measuremnets
    detect_thresh = 4.   #threshold for object detection

    ''' get list of image names to process'''       
    
    if RCDfiles == True: # Option for RCD or fits import - MJM 20210901
        imagePaths = sorted(minuteDir.glob('*.rcd'))
    else:
        imagePaths = sorted(minuteDir.glob('*.fits'))  
        
    del imagePaths[0]
    
    field_name = imagePaths[0].name.split('_')[0]  #which of 11 fields are observed
    
    ''' get 2d shape of images, number of image in directory'''
    
    if RCDfiles == True:
        x_length, y_length, num_images = cir.getSizeRCD(imagePaths) 
    else:
        x_length, y_length, num_images = getSizeFITS(imagePaths)

    print (datetime.datetime.now(), "Imported", num_images, "frames")
    
    #check if there are enough images in the current directory 
    minNumImages = len(kernel.array)*3         #3x kernel length
    if num_images < minNumImages:
        print (datetime.datetime.now(), "Insufficient number of images, skipping...")
        return
    

    ''' create star positional data from median combined stack'''
    
    print (datetime.datetime.now(), field_name, 'starfinding...',)
    
    #name of .npy file to save star positions in 
    star_pos_file = base_path.joinpath('ColibriArchive', str(day_stamp), minuteDir.name + '_' + str(detect_thresh) + 'sig_pos.npy')

    # Remove position file if it exists - MJM (modified RAB Feb 2022)
    if star_pos_file.exists():
        star_pos_file.unlink()

    numtoStack = 9          #number of images to include in stack
    startIndex = 1          #which image to start stack at (vignetting in 0th image)

    min_stars = 30          #minimum stars in an image
    star_find_results = []  #to store results of star finding
    
    #run until the minimum number of stars is found, exit if the images in the whole directory are blank
    while len(star_find_results) < min_stars:
        
        
        #print('stacking images %i to %i\n' %(startIndex, numtoStack))
        
        #create median combined image for star finding
        stacked = cir.stackImages(minuteDir, savefolder, startIndex, numtoStack, bias)

        # make list of star coords and half light radii using a conservative
        # threshold scaled to the number of images stacked
        star_find_results = tuple(cp.initialFind(stacked, detect_thresh*numtoStack**0.5))

        #remove stars where centre is too close to edge of frame
        # edge_buffer = 1     #number of pixels between edge of star aperture and edge of image
        # star_find_results = tuple(x for x in star_find_results if x[0] + ap_r + edge_buffer < x_length and x[0] - ap_r - edge_buffer > 0)
        # star_find_results = tuple(y for y in star_find_results if y[1] + ap_r + edge_buffer < x_length and y[1] - ap_r - edge_buffer > 0)
        star_find_results = tuple(x for x in star_find_results if x[0] + Edge_buffer < x_length and x[0] - Edge_buffer > 0)
        star_find_results = tuple(y for y in star_find_results if y[1] + Edge_buffer < x_length and y[1] - Edge_buffer > 0)
        #increase start image counter
        startIndex += 1     #adjust number to start stack if there weren't enough stars
            
        #check if we've reached the end of the minute, return error if so
        if startIndex + 9 >= num_images:
            print(f"no good images in minute: {minuteDir}")
            print (f"{datetime.datetime.now()} Closing: {minuteDir}")
            print ("\n")
            return
        
    #print('star finding file index: ', i)
    
    #save to .npy file
    #star position file format: x  |  y  | half light radius
    np.save(star_pos_file, star_find_results)
        
    #save radii and positions in separate arrays
    star_find_results = np.array(star_find_results)
    radii = star_find_results[:,-1]
    initial_positions = star_find_results[:,:-1]

    num_stars = len(initial_positions)      #number of stars in image
    
    #print(datetime.datetime.now(), 'Done star finding. Number of stars found: ', num_stars) 
    
    ''' Drift calculations '''
    if RCDfiles == True: # Choose to open rcd or fits - MJM
        fframe_data,fframe_time = cir.importFramesRCD(imagePaths, startIndex, 1, bias)   #import first image
        headerTimes = [fframe_time] #list of image header times
        lframe_data,lframe_time = cir.importFramesRCD(imagePaths, len(imagePaths)-1, 1, bias)  #import last image

    else:
        fframe_data,fframe_time = importFramesFITS(imagePaths, startIndex, 1, bias)      #data and time from 1st image
        headerTimes = [fframe_time]  #list of image header times
        lframe_data,lframe_time = importFramesFITS(imagePaths, len(imagePaths)-1, 1, bias) #data and time from last image

    drift_pos = np.empty([2, num_stars], dtype = (np.float64, 2))  #array to hold first and last positions
    drift_times = []   #list to hold times for each set of drifted coords
    GaussSigma = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile
    print(f"Weighting Radius for Starfinding (GaussSigma) = {GaussSigma}")

    #refined star positions and times for first image
    first_drift = cp.refineCentroid(fframe_data,fframe_time[0], initial_positions, GaussSigma)
    drift_pos[0] = first_drift[0]
    drift_times.append(first_drift[1])

    #refined star positions and times for last image
    drift_multiple = 10  # relaxation factor on our centroid finding
    last_drift = cp.refineCentroid(lframe_data,lframe_time[0], drift_pos[0], GaussSigma*drift_multiple)
    drift_pos[1] = last_drift[0]
    drift_times.append(last_drift[1])
    drift_times = Time(drift_times, precision=9).unix

    #get median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = cp.averageDrift(drift_pos[0],drift_pos[1], drift_times[0],drift_times[1])
    
    #check drift rates
    driftTolerance = 2.5e-2   #px per s
    
    if abs(x_drift) > driftTolerance or abs(y_drift) > driftTolerance:
        drift = True # variable to check whether stars have drifted since last frame
    else:
        drift = False

    driftErrorThresh = 1        #threshold for drift that is manageable
    
    #check if stars have drifted too much, return error if so
    if abs(np.median(x_drift)) > driftErrorThresh or abs(np.median(y_drift)) > driftErrorThresh:
        print (datetime.datetime.now(), "Significant drift, skipping ", minuteDir) 
        return 0

    ''' flux and time calculations with optional time evolution '''
    #c3 = timer.time()
    #image data (2d array with dimensions: # of images x # of stars)
    starData = np.empty((num_images, num_stars), dtype=(np.float64, 4))
    
    #get first image data from initial star positions
    starData[0] = tuple(zip(initial_positions[:,0], 
                        initial_positions[:,1], 
                        #sumFlux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                        # (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(),
                        (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(),
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(fframe_time, precision=9).unix)))

    if drift:  # time evolve moving stars
    
        print(f"{minuteDir} Drifted - applying drift to photometry {x_drift} {y_drift}")
        
        # Loop through each image in the minute-long dataset in chunks
        chunk_size = 10
        residual   = (num_images-1)%chunk_size
        for i in range((num_images-1)//chunk_size):
            imageFile,imageTime = cir.importFramesRCD(imagePaths,chunk_size*i+1, chunk_size, bias)
            headerTimes = headerTimes + imageTime
            for j in range(chunk_size):
                starData[chunk_size*i+j+1] = cp.timeEvolve(imageFile[j],
                                                            deepcopy(starData[chunk_size*i+j]),
                                                            imageTime[j],
                                                            ap_r,
                                                            num_stars,
                                                            (x_length, y_length),
                                                            (x_drift, y_drift))
                
                gc.collect()
            
        # Process remaining chunks
        imageFile,imageTime = cir.importFramesRCD(imagePaths,
                                                  num_images-residual,
                                                  residual,
                                                  bias)
        headerTimes = headerTimes + imageTime
        for i in range(residual+1)[:0:-1]:
            starData[num_images-i] = cp.timeEvolve(imageFile[residual-i],
                                                    deepcopy(starData[num_images-i-1]),
                                                    imageTime[residual-i],
                                                    ap_r,
                                                    num_stars,
                                                    (x_length, y_length),
                                                    (x_drift, y_drift))        
        
            
        gc.collect()
        
        # import matplotlib.pyplot as plt
        # for i in range(len(starData[0])):
        #     flux=[]
        #     frame=[]
        #     x_coords_in=starData[0][i][0]
        #     y_coords_in=starData[0][i][1]
        #     x_coords_fi=starData[-1][i][0]
        #     y_coords_fi=starData[-1][i][1]
        #     for j in range(len(starData)):
        #         flux.append(starData[j][i][2])
        #         frame.append(j)
                
        
        #     fig, ax1 = plt.subplots()
        #     ax1.scatter(frame, flux,label="initial pos: x="+str(x_coords_in)+" y="+str(y_coords_in)+"\n final pos: x="+str(x_coords_fi)+" y="+str(y_coords_fi))
        #     plt.legend()
        #     plot_path=base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name)
        #     if not os.path.exists(plot_path):
        #         os.mkdir(plot_path)
        #     plt.savefig(plot_path.joinpath('_star_'+str(i) + '.png'))
            
        #     plt.close()
    
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        
        print(f'{minuteDir} No drift')
        
        # Loop through each image in the minute-long dataset in chunks
        chunk_size = 10
        residual   = (num_images-1)%chunk_size
        for i in range((num_images-1)//chunk_size):
            imageFile,imageTime = cir.importFramesRCD(imagePaths,chunk_size*i+1, chunk_size, bias)
            headerTimes = headerTimes + imageTime
            starData[chunk_size*i+1:chunk_size*(i+1)+1] = cp.getStationaryFlux(imageFile,
                                                                               deepcopy(starData[chunk_size*i]),
                                                                               imageTime,
                                                                               ap_r,
                                                                               num_stars,
                                                                               (x_length, y_length))
                
            gc.collect()
            
        # Process remaining chunks
        imageFile,imageTime = cir.importFramesRCD(imagePaths,
                                                  num_images-(num_images-1)%chunk_size,
                                                  (num_images-1)%chunk_size,
                                                  bias)
        headerTimes = headerTimes + imageTime
        starData[num_images-residual:] = cp.getStationaryFlux(imageFile,
                                                              deepcopy(starData[num_images-residual-1]),
                                                              imageTime,
                                                              ap_r,
                                                              num_stars,
                                                              (x_length, y_length))
                
        gc.collect()
    
    # import matplotlib.pyplot as plt
    # for starNum in range(0,num_stars):
    #     flux=starData[:, starNum, 2]
    #     x_coords_in=starData[0][starNum][0]
    #     y_coords_in=starData[0][starNum][1]
    #     x_coords_fi=starData[-1][starNum][0]
    #     y_coords_fi=starData[-1][starNum][1]
    #     fig, ax1 = plt.subplots()
    #     ax1.scatter(range(0,len(starData)), flux,label="initial pos: x="+str(x_coords_in)+" y="+str(y_coords_in)+"\n final pos: x="+str(x_coords_fi)+" y="+str(y_coords_fi))
    #     plt.legend()
    #     plot_path=base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name)
    #     if not os.path.exists(plot_path):
    #         os.mkdir(plot_path)
    #     plt.savefig(plot_path.joinpath('_star_'+str(i) + '.png'))
        
    #     plt.close()

        # for i in range(len(starData[0])):
        #     flux=[]
        #     frame=[]
        #     x_coords_in=starData[0][i][0]
        #     y_coords_in=starData[0][i][1]
        #     x_coords_fi=starData[-1][i][0]
        #     y_coords_fi=starData[-1][i][1]
        #     for j in range(len(starData)):
        #         flux.append(starData[j][i][2])
        #         frame.append(j)
                
        
        #     fig, ax1 = plt.subplots()
        #     ax1.scatter(frame, flux,label="initial pos: x="+str(x_coords_in)+" y="+str(y_coords_in)+"\n final pos: x="+str(x_coords_fi)+" y="+str(y_coords_fi))
        #     plt.legend()
        #     plot_path=base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name)
        #     if not os.path.exists(plot_path):
        #         os.mkdir(plot_path)
        #     plt.savefig(plot_path.joinpath('_star_'+str(i) + '.png'))
            
        #     plt.close()
    

    #d3 = timer.time() - c3
    # data is an array of shape: [frames, star_num, {0:star x, 1:star y, 2:star flux, 3:unix_time}]

    #print (datetime.datetime.now(), 'Photometry done.')

    ''' Dip detection '''
   
    #perform dip detection for all stars
    
    #loop through each detected object
    dipResults = np.array([cp.dipDetection(starData[:, starNum, 2], kernel, starNum, sigma_threshold) for starNum in range(0,num_stars)],
                          dtype=object)
   
    # event_frames = dipResults[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    # light_curves = dipResults[:,1]         #array of light curves (empty if no event detected)
    # dip_types = dipResults[:,2]             #array of keywords describing type of dip detected
    
    event_frames = dipResults[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    light_curves = dipResults[:,1]         #array of light curves (empty if no event detected)
    
    conv_flux=dipResults[:,2]              #array of convolved light curves (empty if no event detected)
    lightcurve_std=dipResults[:,3]          #std of minute-long lightcurve
    lightcurve_mean= dipResults[:,4]        #mean of minute-long lightcurve
    Bkg_std= dipResults[:,5]                #std of convolved flux background
    Bkg_mean= dipResults[:,6]               #mean of convolved flux background
    conv_min= dipResults[:,7]               #minimum value of the convolution
    significance=dipResults[:,8]          #significance of the event x*sigma

    ''' data archival '''
    
    secondsToSave =  1.0    #number of seconds on either side of event to save 
    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  #save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  #light curves for each star to be saved
    #save_types = dip_types[np.where(event_frames > 0)]
    
    save_conv_flux=conv_flux[np.where(event_frames > 0)]    #save convolved flux for each event
    save_lc_std=lightcurve_std[np.where(event_frames > 0)]  #save lightcurve std for each event
    save_lc_mean=lightcurve_mean[np.where(event_frames > 0)]#save lightcurve mean
    save_bkg_std=Bkg_std[np.where(event_frames > 0)]        #save background std of convolved lightcurve
    save_bkg_mean=Bkg_mean[np.where(event_frames > 0)]      #save background mean of convolved lightcurve
    save_minVal=conv_min[np.where(event_frames > 0)]        #save minimum value of the convolution
    save_sigma=significance[np.where(event_frames > 0)]     #save significance of each event

    #loop through each detected event
    j=0
    for f in save_frames:
        
        date = headerTimes[f].split('T')[0]                                 #date of event
        time = headerTimes[f].split('T')[1].split('.')[0].replace(':','')   #time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     #coords of occulted star
        mstime = headerTimes[f].split('T')[1].split('.')[1]                 # micros time of event
       # print(datetime.datetime.now(), ' saving event in frame', f)
        
        star_all_flux = save_curves[np.where(save_frames == f)][0]  #total light curve for current occulted star
        star_all_conv=save_conv_flux[np.where(save_frames == f)][0] #total convolved light curve for current occulted star
        #text file to save results in
        #saved file format: 'det_date_time_star#_telescope.txt'

        savefile = base_path.joinpath('ColibriArchive', str(obs_date), ''.join(('det_', date, '_', time, '_', mstime, '_star', str(np.where(event_frames == f)[0][0]), '_', telescope, '.txt')))
        #columns: fits filename and path | header time (seconds) |  star flux
        
        #open file to save results
        # with open(savefile, 'w') as filehandle:
            
        #     #file header
        #     filehandle.write('#\n#\n#\n#\n')
        #     filehandle.write('#    Event File: %s\n' %(imagePaths[f]))
        #     filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
        #     filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f]))
        #     filehandle.write('#    Telescope: %s\n' %(telescope))
        #     filehandle.write('#    Field: %s\n' %(field_name))
        #     filehandle.write('#    Dip Type: %s\n' %(save_types[np.where(save_frames == f)][0]))
        #     filehandle.write('#    Median Flux: %.2f\n' %(np.median(star_all_flux)))
        #     filehandle.write('#    Stddev Flux: %.3f\n' %(np.std(star_all_flux)))
        #     filehandle.write('#\n#\n#\n')
        #     filehandle.write('#filename     time      flux\n')
        
        with open(savefile, 'w') as filehandle:
            
            #file header
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#    Event File: %s\n' %(imagePaths[f]))
            filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
            filehandle.write('#\n')
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f]))
            filehandle.write('#    Telescope: %s\n' %(telescope))
            filehandle.write('#    Field: %s\n' %(field_name))
            filehandle.write('#    significance: %.3f\n' %(save_sigma[j]))
            filehandle.write('#    Raw lightcurve std: %.4f\n' %(save_lc_std[j]))
            filehandle.write('#    Raw lightcurve mean: %.4f\n' %(save_lc_mean[j]))
            filehandle.write('#    Convolution background std: %.4f\n' %(save_bkg_std[j]))
            filehandle.write('#    Convolution background mean: %.4f\n' %(save_bkg_mean[j]))
            filehandle.write('#    Convolution minimal value: %.4f\n' %(save_minVal[j]))
            filehandle.write('#\n#\n')
            filehandle.write('#filename     time      flux     conv_flux\n')
           
            ''' save data '''
            j=j+1
           #if the portion of the light curve is at the beginning, start at 0th image
            if f - save_chunk <= 0:  
        
                files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= 0 and i < f + save_chunk]  #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                         #part of light curve to save
                star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                
                #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                    filehandle.write('%s %f  %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))
            
            #if portion of light curve to save is not at the beginning
            else:
                
                #if the portion of the light curve to save is at the end of the minute, end at the last image
                if f + save_chunk >= num_images:
        
                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk]    #list of filenames to save (RAB 042222)
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                                     #part of light curve to save
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))

                #if the portion of the light curve to save is not at beginning or end of the minute, save the whole portion around the event
                else:  

                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk and i < f + save_chunk]     #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                         #part of light curve to save                    
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))



    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", minuteDir)
    print ("\n")

    #print("Drift calcs",c3)
    
    #update star-minutes
    gc.collect()
    return num_stars*num_images
    

#------------------------------------main-------------------------------------#


if __name__ == '__main__':
    
    
###########################
## Argument Parser Setup
###########################

    ## Generate argument parser
    arg_parser = argparse.ArgumentParser(description="First-level data analysis for the Colibri telescopes",
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    ## Available argument functionality
    arg_parser.add_argument('path', help='Path to base directory')
    arg_parser.add_argument('date', help='Observation date (YYYY/MM/DD) of data to be processed')
    arg_parser.add_argument('-s', '--sigma', help='Sigma threshold', default=6.0,type=float)
    arg_parser.add_argument('-R','--RCD', help='Read RCD files directly (otherwise convert to .fits)', action='store_false')
    arg_parser.add_argument('-p', '--noparallel', help='Disable parallelism, run in sequence instead', action='store_false')
    arg_parser.add_argument('-g', '--lowgain', help='Analyze low-gain images', action='store_false')
    arg_parser.add_argument('-t', '--test', help='Test functionality for Peter Quigley', action='store_true')
    
    ## Process argparse list as useful variables
    cml_args = arg_parser.parse_args()
    
    if cml_args.test:
        # Processing parameters
        RCDfiles,gain_high = True,True
        runPar = cml_args.noparallel
        sigma_threshold = cml_args.sigma
        
        # Target data
        base_path = pathlib.Path('/', 'home', 'pquigley', 'ColibriRepos')
        obs_date = datetime.date(2022, 8, 12)
        
    else:
        # Processing parameters
        RCDfiles = cml_args.RCD
        runPar = cml_args.noparallel
        gain_high = cml_args.lowgain
        sigma_threshold = cml_args.sigma
        
        # Target data
        base_path = pathlib.Path(cml_args.path) # path to data directory
        obsYYYYMMDD = cml_args.date # date to be analyzed formatted as "YYYY/MM/DD"
        obsdatesplit = obsYYYYMMDD.split('/')
        obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
        

    ## Get name of telescope
    try:
        telescope = os.environ['COMPUTERNAME']
    except KeyError:
        telescope = "TEST"
        
    
###########################
## Pre-Analysis Preparation
###########################
    
    '''get filepaths'''
 
    #directory for night-long dataset
    night_dir = base_path.joinpath('ColibriData', str(obs_date).replace('-', ''))      #path to data
 
    #subdirectories of minute-long datasets (~2400 images per directory)
    minute_dirs = [f for f in night_dir.iterdir() if f.is_dir()]  
    
    #parent directory of bias images
    bias_dir = [f for f in minute_dirs if 'Bias' in f.name][0]
    
    #remove bias directory from list of image directories and sort
    minute_dirs = [f for f in minute_dirs if 'Bias' not in f.name]
    minute_dirs.sort()
    
    print ('folders', [f.name for f in minute_dirs])
         
    
    '''get median bias image to subtract from all frames'''
    
    NumBiasImages = 9       #number of bias images to combine in median bias images

    #get 2d numpy array with bias datetimes and master bias filepaths
    MasterBiasList = cir.makeBiasSet(bias_dir, base_path, obs_date, NumBiasImages)
    

    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel


    ''''run pipeline for each folder of data'''
 

###########################
## Run in Parallel
###########################   
 
    #running in parallel (minute directories are split up between cores)
    if runPar == True:
        print('Running in parallel...')
        start_time = timer.time()
        finish_txt=base_path.joinpath('ColibriArchive', str(obs_date), telescope+'_done.txt')
        
        pool_size = multiprocessing.cpu_count() - 2
        pool = Pool(pool_size)
        args = ((minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold,
                 base_path,obs_date,telescope,RCDfiles,gain_high) for f in range(len(minute_dirs)))
        
        try:
            star_minutes = pool.starmap(firstOccSearch,args)
            end_time = timer.time()
            starhours = sum(star_minutes)/(2399*60.0)
            print(f"Calculated {starhours} star-hours\n", file=sys.stderr)
            print(f"Ran for {end_time - start_time} seconds", file=sys.stderr)
            
            with open(finish_txt, 'w') as f:
                f.write(f'Calculated {starhours} star-hours\n')
                f.write(f'Ran for {end_time - start_time} seconds')
        except:
            logging.exception("failed to parallelize")
            
        pool.close()
        pool.join()


###########################
## Run in Sequence 
###########################  

    else:
        #raise NotImplementedError("Not running in parallel needs maitenance.\nSorry for the inconvenience!\n-Peter Q (2022/11/05)")
        
        # Added a check to see if the fits conversion has been done. - MJM 
        #only run this check if we want to process fits files - RAB
        if RCDfiles == False:
                
            #check if conversion indicator file is present
            if not minute_dirs[f].joinpath('converted.txt').is_file():
                        
                print('Converting to .fits')
                
                #make conversion indicator file
                with open(minute_dirs[f].joinpath('converted.txt'), 'a'):
                    os.utime(str(minute_dirs[f].joinpath('converted.txt')))
                            
                #do .rcd -> .fits conversion using the desired gain level
                if gain_high:
                    os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]) + ' high')
                else:
                    os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]))
                    
        else:
            print('Already converted raw files to fits format.')
            print('Remove file converted.txt if you want to overwrite.')
        
        for f in range(len(minute_dirs)):
           


            print(f"Running on... {minute_dirs[f]}")

            start_time = timer.time()

            print('Running sequentially...')
            firstOccSearch(minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold,
                           base_path,obs_date,telescope,RCDfiles,gain_high)
            
            gc.collect()

            end_time = timer.time()
            print(f"Ran for {end_time - start_time} seconds", file=sys.stderr)

#           with open("logs/timing.log","a") as f:
#               f.write(f"Ran for {end_time - start_time} seconds\n\n")


