import logging
import os  # for correctly handling the directory position
from scipy import io  # this is needed to load the settings of xtomo
import numpy as np
import MDSplus
import xarray as xray  # this is needed as tdi save into an xray
import copy
log = logging.getLogger(__name__)  # pylint: disable=invalid-name

class AXUV(object):

    def __init__(self, shot, LoS=None, trange=None):
        """
        Programs to load the calibrated data of AXUV
        diagnostic together with appropriate geometrical
        information

        Parameters
        ----------
        Shot
            Shot number
        LoS
            Int or list. Optional, eventually load only given LoS
        trange
            Optional time range to read the data in a
            given time range
    
        Returns
        -------
        Stored Data are save in axuv.AXUV.Data as an xarray DataSet 
        containing also the geometrical information relative to the
        
        """

        # define the shot
        self.shot = shot
        self.LoS = LoS
        self.trange=trange
        if self.LoS is not None:
            self.LoS = np.atleast_1d(self.LoS).astype('int')-1
        # restore the geometry of the AXUV
        _pathFile = os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)), 'axuvcalib', 'axuv.mat')
        self.calibration = io.loadmat(_pathFile)
        self._patchCalibration()
        # define the patch panel for the 'Plasma' array
        bolomap = np.stack((
            [1, 20, 2, 19, 3, 18, 4, 17, 5, 16, 6, 15, 7, 14, 8, 13, 9, 12, 10, 11],
            [43, 62, 44, 61, 45, 60, 46, 59, 47, 58, 48, 57, 49, 56, 50, 55, 51, 54, 52, 53],
            [1, 94, 2, 93, 3, 92, 4, 91, 5, 90, 6, 89, 7, 88, 8, 87, 9, 86, 10, 85],
            [43, 42, 44, 41, 45, 40, 46, 39, 47, 38, 48, 37, 49, 36, 50, 35, 51, 34, 52, 33],
            [85, 84, 86, 83, 87, 82, 88, 81, 89, 80, 90, 79, 91, 78, 92, 77, 93, 76, 94, 75],
            [21, 42, 22, 41, 23, 40, 24, 39, 25, 38, 26, 37, 27, 36, 28, 35, 29, 34, 30, 33],
            [65, 84, 66, 83, 67, 82, 68, 81, 69, 80, 70, 79, 71, 78, 72, 77, 73 ,76, 74, 75])).transpose()
        
        self.boards = np.stack((np.tile(11, 20),
                                np.tile(11, 20),
                                np.tile((12, 11), 10),
                                np.tile(12, 20),
                                np.tile(13, 20),
                                np.tile(13, 20),
                                np.tile(13, 20))).transpose()
        # there are some special cases we need to take into account
        if 44499 <= self.shot < 50000:
            bolomap[:, [2, 3, 4]] = np.flipud(bolomap[:, [2, 3, 4]])
            self.boards[:, [2, 3, 4]] = np.flupud(self.boards[:, [2, 3, 4]])
            self.gains = 400e3
        elif self.shot >= 50000:
            bolomap[:, 2] = [21, 20, 22, 19, 23, 18, 24, 17, 25 ,16, 26, 15, 27, 14, 28, 13, 29 ,12, 30, 11]
            bolomap[:, 3] = [65, 62, 66, 61, 67, 60, 68, 59, 69, 58, 70, 57, 71, 56, 72, 55, 73, 54, 74, 53]
            bolomap[:, 4] = [11, 10, 12, 9, 13, 8, 14, 7, 15, 6, 16, 5 , 17, 4, 18, 3, 19, 2, 20, 1]
            self.boards[:, 2] = np.tile(12, 20)
            self.boards[:, 3] = np.tile(12, 20)
            self.boards[:, 4] = np.tile(13, 20)
            self.gains = 400e3
        else:
            log.warn('Too old shot for this class')
            sys.exit(0)

        self.channel= bolomap.reshape(140, order='F')
        self.boards = self.boards.reshape(140, order='F')
        self._axuvTree = MDSplus.Tree('axuv', self.shot)
        self._check_dtaq()
        self._axuvTree.quit()
        self._Tree = MDSplus.Tree('tcv_shot', self.shot)
        # now we call the reading function
        self._getdata()
        self._Tree.quit()
        
    def _patchCalibration(self):
        """
        Apply few patches to the calibration files since
        there are few things which are not needed
        """
        # this is restore from a mat file we need to drop few
        # keys which are not useful
        self.calibration.pop('__header__', None)
        self.calibration.pop('__globals__', None)
        self.calibration.pop('__version__', None)
        self.calibration.pop('majorcamlim', None)
        self.calibration['slitheight'] = self.calibration['slitheight'].transpose()
        # then squeeze all the variables which are 1D
        for key in self.calibration.keys():
            if (key != 'xchord') & (key != 'ychord'):
                self.calibration[key] = self.calibration[key].squeeze()
        
    def _check_dtaq(self):
        """
        Given the channels and board check for acquired signal.
        Eventually limit to the desired LoS including also
        a check on the dictionary 
        
        """
        lch = []
        for board in ('11', '12', '13'):
            lch.append(
                MDSplus.Data.compile(
                    'getnci(".BOARD_0'+board+':CHANNEL_*", "LENGTH")').evaluate().data())
        bg = np.stack((np.tile(11, 96), np.tile(12, 96), np.tile(13, 96)))
        chg = np.stack((np.arange(96), np.arange(96), np.arange(96)))
        # determine if we have channels
        if 0 < np.where(lch == 0)[0].size < lch.size:
            self.channel[np.where(lch ==0)[0]] = np.nan
            self.boards[np.where(lch == 0)[0]] = np.nan
            log.warn(' %03i Channels not acquired in this shot ' % np.where(lc==0).size)
        elif np.where(lch == 0)[0].size == len(lch):
            log.warn('No data for this shots ')
            sys.exit(0)
        else:
            log.warn('We have all the channels for shot %5i' % self.shot)
        # now in case we are only dealing with few LoS we limit
        # to these channels and boards
        if self.LoS is not None:
            self.channel = self.channel[self.LoS]
            self.boards = self.boards[self.LoS]
            for key in self.calibration.keys():
                if key == 'xchord':
                    self.calibration[key] = self.calibration[key][:, self.LoS]
                elif key == 'ychord':
                    self.calibration[key] = self.calibration[key][:, self.LoS]
                else:
                    self.calibration[key] = self.calibration[key][self.LoS]
                    
            
    def _getdata(self):
        """
        Get the appropriate calibrated data

        """

        data = []
        for channel, board in zip(self.channel, self.boards):
            data.append(self._Tree.getNode(
                r'\atlas::dt196_axuv_00' + board.astype('string')[1] +
                ':CHANNEL_{:03}'.format(channel)).data())

        data = np.asarray(data)
        time = self._Tree.getNode(r'\atlas::dt196_axuv_00' +
                                  self.boards[0].astype('string')[1] +
                                  ':CHANNEL_{:03}'.format(self.channel[0])).getDimensionAt().data()
        if self.LoS is None:
            Data = xray.DataArray(data,
                                  coords=[np.linspace(1, 140, 140, dtype='int'), time],
                                  dims=['LoS', 'time'])
        else:
            Data = xray.DataArray(data, coords=[self.LoS+1, time], dims=['LoS', 'time'])
            
        Data -= Data.where(Data.time <= -0.01).mean(dim='time')            
        self.Data = Data/0.24/self.gains/np.expand_dims(
            self.calibration['etend'], axis=1)
        for key in self.calibration.keys():
            self.Data.attrs[key] = self.calibration[key]

        # in case we are limiting to a given number in trange than we
        # apply a limit
        if self.trange is not None:
            self.Data = self.Data.where(((Data.time >= self.trange[0]) &
                                         (Data.time <= self.trange[1])),
                                        drop=True)
            
        return self.Data
