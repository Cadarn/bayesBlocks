import pandas as pd
import numpy as np
import pylab as plt
'''
Implementing Bayesian Blocks based upon the paper below:
==========================================================================
 Title: Studies in Astronomical Time Series Analysis. 
        VI. Bayesian Block Representations
 Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
==========================================================================

'''

def read_ttedata(filename):
    '''Test data is taken as BATSE tte ASCII file sourcesd from:
    ftp://legacy.gsfc.nasa.gov/compton/data/batse/ascii_data/batse_tte/
    
    Inputs:     filename - full path to BATSE TTE data file
    
    Outputs:    
    
    '''
    
    #Extract out the header info from the file
    fileIn = open(filename, 'r')
    header = []
    for i in range(5):
        header.append(fileIn.readline())
        
    #Number of data points is contained in 2nd element of header
    npts = int(header[1].split('=')[1])
    
    #Now read in rest of data
    data = fileIn.readlines()
    data2 = []
    for line in data:
        temp = line.split()
        data2 = data2+temp
    #Convert all the strings into ints
    data = [int(x) for x in data2]
    fileIn.close()
    
    #construct dataframe
    df = pd.DataFrame({'TIME' : data[0:npts-1],
                       'CHANNEL' : data[npts:2*npts-1],
                       'DETECTOR' : data[2*npts: 3*npts-1]
                        })
    #Convert time to seconds
    df.TIME = df.TIME * 1e-6                    
    return df
    
def find_blocks(df):
    '''Find bayesian block representation of TTE data
    '''
    data_mode = 1 #Represents time tagged event data (TTE)
    
    tt = np.array(df.TIME)
    num_points = len(df)
    dt = np.diff(tt, n=1)
    nn_vec = np.ones(num_points)
    dt_median = np.median(dt)
    
    #Set false positive rate
    fp_rate = 1 - 0.95
    
    #Set NCP prior
    ncp_prior = 4 - np.log(fp_rate/(0.0136*num_points**(0.478)))
    
    #No iteration invoked
    tt_start = tt[0] - 0.5*dt_median
    tt_stop = tt[-1] + 0.5*dt_median
    
    #No for the work
    change_points = []
    count_vec = []
    #Setup block lengths
    tempTime = 0.5*(tt[1:] + tt[0:-1])
    tempTime = np.append(tempTime, tt_stop)
    tempTime = np.insert(tempTime, 0, tt_start)
    block_length = tt_stop - tempTime
    print('Block Length', block_length)
    
    while 1:
        best = np.array([])
        last = np.array([])
        
        for R in range(num_points):
            arg_log = block_length[0:R+1] - block_length[R+1]
            arg_log[arg_log == 0.] = np.inf
            nn_cum_vec = np.cumsum(nn_vec[R::-1])
            nn_cum_vec = nn_cum_vec[R::-1]
            
            fit_vec = nn_cum_vec * (np.log(nn_cum_vec) - np.log(arg_log))
            
            temp = np.concatenate((np.array([0]), best))+ fit_vec - ncp_prior
            best = np.append(best, np.max(temp))
            last = np.append(last, np.argmax(temp))
            
        #Now find change points by iteratively peeling off the last block
        print('%i unique change points identified' %len(np.unique(last)))
        index = last[-1]
        change_points = []
    
        while index > 0:
            change_points.append(index)
            index = last[index-1]
        break
    
    change_points = change_points[::-1]                
    num_changepoints = len(change_points)
    num_blocks = num_changepoints + 1
    
    rate_vec = np.zeros(num_blocks)
    num_vec =  np.zeros(num_blocks)
    dt_vec = np.zeros(num_blocks)
    tt_1_vec = np.zeros(num_blocks)
    tt_2_vec = np.zeros(num_blocks)
    
    cpt_use = [0] + change_points
    
    for id_block in range(num_blocks):
        ii_1 = cpt_use[id_block]   #Start
        if id_block < num_blocks -1 :
            ii_2 = cpt_use[id_block + 1] - 1
        else:
            ii_2 = num_points - 1
            
        if data_mode == 3:
            pass
        else:
            num_this = np.sum(nn_vec[ii_1:ii_2])
            delta_tt = tt[ii_2] - tt[ii_1]
            num_vec[id_block] = num_this
            rate_this = num_this/delta_tt
            rate_vec[id_block] = rate_this
            
    return change_points, num_vec, rate_vec, best, last, ncp_prior, nn_vec
    
class bblock(object):
    '''Bayesian Blocks Analysis class
    
    Author: Adam Hill   Date: 09/05/2014
    Implementing the algorithm described by:
    ==========================================================================
    Title: Studies in Astronomical Time Series Analysis. 
        VI. Bayesian Block Representations
    Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
    ==========================================================================
    '''
    def __init__(self, data, data_mode=1):
        '''Initialisation of bblock object.
           Inputs:
               data -> pandas dataframe containing time series
               data_mode -> Form of data: 1 = time tagged events; 2 = binned data; 3 = point measurements (Default: 1)
        '''
        self.data = data
        self.dataMode = data_mode
        self.npoints = len(self.data)
        description = "Object representing a Bayesian Blocks representation of data"
        author = "Coded by ABH 09/05/2014"

        
    def find_blocks(self, fp_rate = 0.003):
        '''Method to run the top-level analysis to find the bayesian block segmentation
        Input:
            fp_rate -> Prob that the change is not real. Default: 0.003 i.e. 3 sigma
        '''
        #Process the input data to create data structures needed to do calculations
        self._ProcessData()
        #Calculate the prior on the change points
        self._CalcPrior(fp_rate=fp_rate)
        #Calculate the fitness function and find the change points that maximise it
        self._CalcFitness()
        #Recover the changepoints
        self._RecoverCP()
        #Post process the data
        self._ProcessBlocks()
        
        print('==============')
        print('Analysis of input data has found that it is optimally segemented into %i blocks' %self.num_blocks)

    def _ProcessData(self):
        '''Process the incoming data frame to establish the initial block edges and rate vectors
        '''
        
        #Assign relative exposure of each data point if provided otherwise set all to 1
        if hasattr(self.data, 'EXPOSURE'):
            self.relExp = self.data.EXPOSURE/max(self.data.EXPOSURE)
        else:
            self.relExp = np.ones(self.npoints, dtype=np.float64)
        
        if self.dataMode == 1: #TTE data
            #Need to set times to the arrival times of photons and attribute one photon per bin
            self.time = np.array(self.data.TIME)
            self.nn_vec = np.ones(self.npoints)
        elif self.dataMode == 2: #Binned Data
            self.time = np.array(self.data.TIME)
            if hasattr(self.data, 'COUNTS'):
                self.nn_vec = self.data.COUNTS
            elif hasattr(self.data, 'FLUX') and hasattr(self.data, 'EXPOSURE'):
                self.nn_vec = self.data.FLUX * self.data.EXPOSURE
                #Correct relative exposure array to be uniform now
                self.relExp = np.ones(self.npoints, dtype=np.float64)
            else:
                raise AttributeError("No COUNT column or FLUX & EXPOSURE column in input dataframe")            
        elif self.dataMode == 3: #Point measurements
            self.time = np.arange(self.npoints) #Nominal equal spaced time bins
            try:
                self.nn_vec = self.data.FLUX
            except AttributeError:
                print('No FLUX column provided in input dataframe with DATA_MODE = 3')
                return False
            try:
                self.nn_vec_err = self.data.ERROR
            except AttributeError:
                print('No ERROR column provided in input dataframe with DATA_MODE = 3')
                return False
        else:
            print('No valid data mode found: .dataMode should be 1,2 or 3')
        
        #Now we have a time vector and a 'flux' vector
        #Calculate dt
        dt = np.diff(self.time, n=1)
        #Check all times are in increasing order
        assert min(dt) >= 0, "TIME points are not in increasing order: Check input dataframe"
        
        #Calculate median of dt
        dt_median = np.median(dt)
        #Establish start and end times of observation
        if hasattr(self.data, 'tStart'):
            tStart = self.data.tStart
        else:
            tStart = self.time[0] - 0.5*dt_median #Default start time
        if hasattr(self.data, 'tStop'):
            tStop = self.data.tStop
        else:
            tStop = self.time[-1] + 0.5*dt_median #Default end time
            
        #Finally calculate the block lengths if in data mode 1 or 2
        if self.dataMode != 3:
            tempTime = 0.5*(self.time[1:] + self.time[0:-1])
            tempTime = np.append(tempTime, tStop)
            self.tt = np.insert(tempTime, 0, tStart)
            self.block_length = tStop - self.tt
        print('Input data processed ready for Bayesian Block calculation ...')       

    def _CalcPrior(self, fp_rate = 0.003):
        '''Calculate the change point prior based upon the formulae from section 3 of Scargle et al. 2012.
        ncp_prior is also log(gamma) in the text.
        Input:
            fp_rate -> Prob that the change is not real. Default: 0.003 i.e. 3 sigma
        '''
        self.fp_rate = fp_rate
        if self.dataMode == 3:
            #For Point measurements
            self.ncp_prior = 1.32 + 0.577*np.log10(self.npoints)
        else:
            #For Event Data and Binned Data
            #self.ncp_prior = 4 - 73.53*fp_rate*self.npoints**(-0.478)
            self.ncp_prior = 4 - np.log(fp_rate/(0.0136*self.npoints**(0.478)))
        print('Using a FAP of %f equates to a changepoint prior of %f' %(self.fp_rate, self.ncp_prior))

    def _CalcFitness(self):
        '''Calculate the "fitness" over the variety of bin widths and and record the changepoints'''
        #Arrays to store data as we iterate            
        best = np.array([])
        last = np.array([], dtype=np.int64)
        for j in range(self.npoints):
            #Loop over all the data points and compute the width and index of the final bin for
            #all locations of the j-th changepoint
            if self.dataMode == 3:
                ## Analysis process for point measurements hence Gaussian errors
                sum_x_0 = 0.5*np.cumsum(1/(self.nn_vec_err[j::-1]**2))
                sum_x_1 = np.cumsum(self.nn_vec[j::-1]/(self.nn_vec_err[j::-1]**2))
                fit_vec = (sum_x_1[j::-1]**2.)/(4.*sum_x_0[j::-1])
            else:
                ## Analysis process for event or binned data hence Poisson errors
                arg_log = self.block_length[0:j+1] - self.block_length[j+1]
                #In case any arg_log values equal zero change the value to infinity
                arg_log[arg_log == 0.] = np.inf 
                #Now see how many counts occur in the range of blocks
                nn_cum_vec = np.cumsum(self.nn_vec[j::-1])
                nn_cum_vec = nn_cum_vec[j::-1]
                
                #Now evaluate the fitness function       
                fit_vec = nn_cum_vec * (np.log(nn_cum_vec) - np.log(arg_log))
                
            #Subtract off the prior for number of change_points
            fit_vec -= self.ncp_prior
            fit_vec = np.concatenate((np.array([0]), best))+ fit_vec
            #Store the value and index of the point that maximises the fit function
            best = np.append(best, np.max(fit_vec))
            last = np.append(last, np.int64(np.argmax(fit_vec)))
                                    
        #Store the best and last arrays to the object
        self.best = best
        self.last = last
        print('Block fitness function assessed ...')                        
                                                                        
    def _RecoverCP(self):
        '''Find change points by iteratively peeling off the last block'''
        #Create change_point list and initialise index of cp
        change_points = []
        #First change point index will be the last element of the LAST array
        try:
            cpInd = self.last[-1]
        except NameError:
            print('Trying to run before calculating block fitness: try running _CalcFitness')
            return None
        
        #Keeping looping through array until the index of the change point hits 0
        while cpInd > 0:
            #Store change point
            change_points.append(cpInd)
            #Jump back to the next change point
            cpInd = self.last[cpInd - 1]
            
        #Need to reverse the change_point order  
        change_points = change_points[::-1]
        #Need to add the first point to the list of change points
        change_points.insert(0,0)

        #Set internal parameters        
        self.change_points = np.array(change_points)
        self.num_cp = len(change_points)
        self.num_blocks = len(change_points) + 1
    
        print('Changepoints recovered ...')               

    def _ProcessBlocks(self):
        '''Take the calculated changepoints and process the data to give useful outputs e.g. block rates
        '''
        try:
            assert self.num_cp == len(self.change_points)
        except NameError:
            print('Trying to post-process data prior to calculating blocks: try running _RecoverCP')
            return False
        #If last change point doesn't correspond to end of data then add that as a point
        if self.change_points[-1] != self.npoints - 1:
            self.change_points = np.append(self.change_points, np.array([self.npoints -1]))
        else:
            self.num_blocks -= 1
            
        self.rate_vec = np.zeros(self.num_blocks)
        self.num_vec =  np.zeros(self.num_blocks)
        self.dt_vec = np.zeros(self.num_blocks)
    
        for id_block in range(self.num_blocks):
            ii_1 = self.change_points[id_block]   #Start
        
            if id_block < self.num_blocks - 2 :
                ii_2 = self.change_points[id_block + 1] 
            else:
                ii_2 = self.npoints - 1
            #Check that ii_1 != ii_2:
            if ii_1 == ii_2:
                break #Instance of last change_point matching the last data point
            if self.dataMode == 3:
                xx_this = self.nn_vec[ii_1:ii_2]
                wt_this = 1./(self.nn_vec_err[ii_1:ii_2]**2.)
                self.rate_vec[id_block] = np.sum(wt_this*xx_this)/np.sum(wt_this) 
            else:
                num_this = np.sum(self.nn_vec[ii_1:ii_2])
                delta_tt = self.time[ii_2] - self.time[ii_1]
                self.num_vec[id_block] = num_this
                rate_this = num_this/delta_tt
                self.rate_vec[id_block] = rate_this
                
        print('Post processing complete...')

class bblock_multi(object):
    '''Multi observation Bayesian Blocks Analysis class
    
    Author: Adam Hill   Date: 28/10/2014
    Implementing the algorithm described by:
    ==========================================================================
    Title: Studies in Astronomical Time Series Analysis. 
        VI. Bayesian Block Representations
    Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
    ==========================================================================
    '''
    def __init__(self, time_series_list, data_modes):
        '''Initialisation of bblock_multi object.
           Inputs:
               time_series_list -> A python list of pandas dataframes containing time series
               data_modes -> Either a list of data modes of same length as time_series_list or an integer representing the
                             the data mode for all of the time_series.
                             Form of data: 1 = time tagged events; 2 = binned data; 3 = point measurements (Default: 1)
        '''
        if type(data_modes) == list:
            try:
                assert len(time_series_list) == len(data_modes)
            except AssertionError:
                print('Number of data modes does not match number of time series.')
        else:
            try:
                assert type(data_modes) == int
                data_modes = [data_modes for x in time_series_list]
            except AssertionError:
                print('INVALID data_modes supplied: Should be either a list of data modes or an integer.')

        #Store the original light curves and the Bayesian Blocks representations of each dataset
        self.datasets = []
        self.bbData = []
        for i, lc in enumerate(time_series_list):
            self.datasets.append(lc)
            tempBlocks = bblock(lc, data_mode=data_modes[i])
            tempBlocks.find_blocks()
            self.bbData.append(tempBlocks)
            
        self.data_mode_vec = data_modes
        self.nseries = len(self.datasets)
        self.ncp_prior_vec = np.zeros(self.nseries)
        self.tt_start_vec = np.zeros(self.nseries)
        self.tt_stop_vec = np.zeros(self.nseries)
        self.ii_start_vec = np.zeros(self.nseries)
        
        description = "Object representing a multi-variate Bayesian Blocks representation of multi observation data"
        author = "Coded by ABH 10/11/2014"
        
    def _processTimeMarkers(self):
        '''Process the time markers for each time series'''
        #Set internal time tag array
        self.tt = np.array([])
        ii_start = 0
        row_count = 0
        for id_series in range(self.nseries):
            row_count += 2
            tt_this = self.bbData[id_series].data.TIME
            dt_this = np.diff(tt_this, n=1)
            num_points_this = len(tt_this)
            self.tt = np.append(self.tt, tt_this)
            self.ii_start_vec[id_series] = ii_start
            ii_start += num_points_this
            tt_start = min(self.bbData[id_series].data.TIME) - 0.5*np.median(dt_this)
            tt_start_vec[id_series] = tt_start
            tt_stop = max(dataList[id_series].data.TIME) + 0.5*np.median(dt_this)
            tt_stop_vec[id_series] = tt_stop
            ncp_prior = dataList[id_series].ncp_prior
            ncp_prior_vec[id_series] = ncp_prior            
        
'''
tt = []
ii_start = 0
row_count = 0
for id_series in range(num_series):
    row_count += 2
    data_mode_vec[id_series] = 3
    tt_this = dataList[id_series].data.TIME
    dt_this = np.diff(tt_this, n=1)
    num_points_this = len(tt_this)
    tt = list(tt) + list(tt_this)
    ii_start_vec[id_series] = ii_start
    ii_start += num_points_this
    tt_start = min(dataList[id_series].data.TIME) - 0.5*np.median(dt_this)
    tt_start_vec[id_series] = tt_start
    tt_stop = max(dataList[id_series].data.TIME) + 0.5*np.median(dt_this)
    tt_stop_vec[id_series] = tt_stop
    ncp_prior = dataList[id_series].ncp_prior
    ncp_prior_vec[id_series] = ncp_prior            
'''                                    
                                    
def testBayes():
    '''Test of TTE event analysis with Bayesian blocks'''
    filename = './testData/tteascii.00551'
    df = read_ttedata(filename)
    subDF = df[df.TIME<=0.6]
    
    myBlocks = bblock(subDF, data_mode=1)
    myBlocks.find_blocks(fp_rate = 0.05)
    print(myBlocks.change_points)
    H1 = plt.hist(subDF.TIME, bins=32, histtype='stepfilled', alpha=0.2, normed=True)
    H2 = plt.hist(subDF.TIME, bins=subDF.TIME[myBlocks.change_points], color='k', histtype='step', normed=True)
    plt.show()
    return subDF, myBlocks

    
    
