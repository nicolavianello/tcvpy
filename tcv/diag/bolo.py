"""
BOLO diagnostic

Written by Nicola Vianello
"""

import numpy
from scipy import signal
from scipy import io
import os
import tcv
import xarray as xray


class Bolo(object):
    """
    Load calibrated Bolometry cameras, checking also for good
    not saturated signals
    All methods implemented as staticmethods so that can be called directly
    """

    @staticmethod
    def fromshot(shot, offset=True, filter='gottardi',
                 Los=None):
        """
        Return the calibrated signal of the BOLO diagnostic with
        the possibility to choose given LoS.
        So far it mimic only one of the filter used

        Parameters:
        ----------
        shot: int
            Shot Number
        offset: Boolean. Default True
            If set it remove the offset before the shot.
        filter: String
            Type of available filter for signal processing.
            Available filters are\ ``bessel``\ or\ ``gottardi``\.
            Default is\ ``gottardi``\.x
        Los: Int or list
            Defining which chord to load. If not given collect all
            the 64 channels

        Returns:
        -------
        Calibrated BOLO signals in the form of xray-DataArray
        Attributes:
        -------
        shot: int
            Shot Number
        units: String
            Unit of string
        xchord: float 2D
            LoS start and end radial position
        ychord: float 2D
            LoS start and end vertical position
        angle: float
            Cord angle
        xPO: float
            Radial position of the pinhole cameras
        zPO: float
            vertical position of the pinhole cameras
        Examples:
        -------
        >>> import tcv
        >>> boloData = tcv.diag.Bolo.fromshot(50766, Los=[1, 4, 6])

        """
        conn = tcv.shot(shot)
        # collect the raw data
        raw = conn.tdi(r'\base::bolo:source', dims=('time', 'los'))
        # we lack the correct time bases so we load it
        # and substitue in the XRAY data set
        time = conn.tdi(r'dim_of(\base::bolo:signals)').values
        raw.time.values = time
        # collect the gains
        gains = conn.tdi(r'\base::bolo:gains')
        # collect the calibration
        calibration = conn.tdi(r'\base::bolo:calibration')
        # collect the etendue
        etendue = conn.tdi(r'\base::bolo:geom_fact')
        # define the conversion factor
        convfact = 9.4 * 0.0015 * 0.004
        # collect tau
        tau = conn.tdi(r'\base::bolo:tau')
        # collect the geometry dictionary which will be used afterward.
        # I will append it to the data as additional
        # dictionary
        geodict = Bolo.geo(shot, los=Los)
        # we define the point when the offset removing
        # mechanism is switched off
        start = conn.tdi('timing("401")').values
        if start < 0 and offset:
            raw -= raw.where(((raw.time < 0) &
                              (raw.time > 0.7 * start))).mean(dim='time')
            print ' -- Offset removal -- '
        else:
            print ' -- Offset not removed  --'

        # now we compute the calibrated data
        if filter == 'gottardi':
            ts, sm, smd = Bolo.gottardifilt(raw)
        elif filter == 'savitzky':
            sm, smd = Bolo.savitzkyfilt(raw)
        elif filter == 'bessel':
            sm, smd = Bolo.bessel(raw)
        else:
            print('Filter not implemented, using default gottardi')
            filter = 'gottardi'
            ts, sm, smd = Bolo.gottardifilt(raw)

        dynVolt = sm + smd * tau.values.reshape(1, tau.shape[0])
        data = dynVolt / (gains.values.reshape(1, gains.shape[0]) *
                          calibration.values.reshape(1,
                                                     calibration.shape[0]) *
                          etendue.values.reshape(1,
                                                 etendue.shape[0]) * convfact)
        # transform data on xray data source and adding geo as a dictionary
        out = xray.DataArray(data, dims=('time', 'los'))
        if filter == 'gottardi':
            out.coords['time'] = ts
        else:
            out.coords['time'] = time
        out.coords['los'] = numpy.linspace(1, 64, 64, dtype='int')
        # adding the attributes
        out.attrs['shot'] = shot
        out.attrs['units'] = 'W/m^2'
        # non in case it is called with the LOS we select the appropriate
        # ones. The Geodictionary is already limited
        for key in geodict.keys():
            out.attrs[key] = geodict[key]
        if Los is None:
            return out
        else:
            print ' -- selecting chords -- '
            out2 = out.sel(los=Los)
            return out2

    @staticmethod
    def geo(shot, los=None):
        """

        Parameters
        ----------
        shot: Shot number

        Returns
        -------
        Return a dictionary with all the geometrical
        information for the BOLO diagnostic

        """
        # the position of the pin-hole cameras
        # are given
        xpos = numpy.asarray([0.88, 1.235, 1.235, 1.235,
                              1.235, 1.235, 1.235, 0.88])
        ypos = numpy.asarray([.815, 0.455, 0.455, -0.0025,
                              -.0025, -0.46, -0.46, -0.815])
        base = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'bolocalib')
        geoData = io.loadmat(os.path.join(base,
                                          'geobolo.mat'))
        xchord = geoData['xchord']
        ychord = geoData['ychord']
        angle = geoData['angle'].ravel()
        if los is not None:
            _idxLos = numpy.atleast_1d(los) - 1
            _idxPO = numpy.unique(_idxLos / 8)
            xpos = xpos[_idxPO]
            ypos = ypos[_idxPO]
            xchord = xchord[:, _idxLos]
            ychord = ychord[:, _idxLos]
            angle = angle[_idxLos]

        out = dict([('xPO', xpos), ('zPO', ypos),
                    ('xchord', xchord),
                    ('ychord', ychord),
                    ('angle', angle)])
        return out

    @staticmethod
    def gottardifilt(data, **kwargs):
        """
        Perform an appropriate smoothing of the signal
        according to a given formula
        Parameters.
        ----------
        data: xarray DataSet as obtained from a TDI call
        knoise: number of smoothing iterations
        ibave: Sample no. to calculate the initial value of the signal.
        Default is 80
        ieave: Sample no. to calculate the final value of the signal.
        Default is 10
        alevel: Suppression parameter. Default is 0.16
        tNoise: interval to calculate the derivative noise. Default: [-0.04, 0]

        Returns
        -------
        Smoothed array and time derivative
        """

        # standard definition
        knoise = kwargs.get('knoise', 20)
        ibave = kwargs.get('ibave', 80)
        ieave = kwargs.get('ieave', 10)
        alevel = kwargs.get('alevel', 0.16)
        tNoise = kwargs.get('tNoise', [-0.04, 0])
        time = data.time.values
        # determine the indices where the noise level is computed
        indNoise = ((time >= tNoise[0]) & (time <= tNoise[1]))
        # create a copy of the signal changing the first and last point
        dataC = data.values.transpose().copy()
        dataC[:, 0] = dataC[:, :ibave].mean()
        dataC[:, -1] = dataC[:, (dataC.shape[1] - ieave + 1):-1].mean()
        # copy of the data to perform smoothing
        ys_left = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ys_right = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ys_left[:, 0] = dataC[:, 0]
        ys_right[:, -1] = dataC[:, -1]
        count = 0
        while count < knoise:
            diffnorm = numpy.abs(numpy.diff(dataC, axis=-1) /
                                 numpy.diff(time) /
                                 numpy.max(numpy.abs(
                                     numpy.diff(dataC[:, indNoise],
                                                axis=-1) /
                                     numpy.diff(time[indNoise]))[1:]))

            for k in numpy.arange(1, time.size - 1, 1):
                ys_left[:, k] = (ys_left[:, k - 1] +
                                 dataC[:, k] + dataC[:, k + 1]) / 3. + \
                    numpy.tanh(alevel *
                               numpy.max([diffnorm[:, k - 1],
                                          diffnorm[:, k]])) * \
                    (dataC[:, k] - (ys_left[:, k - 1] +
                                    dataC[:, k] +
                                    dataC[:, k + 1]) / 3.)
                NB = - k
                ys_right[:, NB] = (ys_right[:, NB + 1] +
                                   dataC[:, NB] + dataC[:, NB - 1]) / 3. + \
                    numpy.tanh(alevel *
                               numpy.max([diffnorm[:, NB - 1],
                                          diffnorm[:, NB]])) * \
                    (dataC[:, NB] - (ys_right[:, NB + 1] + dataC[:, NB] +
                                     dataC[:, NB - 1]) / 3.)
            dataC[:, 1:] = (ys_left[:, 1:] + ys_right[:, 1:]) / 2.
            count += 1
        # redefine the time basis
        ts = numpy.zeros(dataC.shape[1])
        ts[:-1] = 0.5 * (time[1:] + time[:-1])
        ts[-1] = ts[-2] + (ts[-2] - ts[-3])
        # define the output smoothed signal
        ys = 0.5 * (dataC + numpy.roll(dataC, 1, axis=-1))
        ys[:, -1] = ys[:, -2]
        # define the output smoothed derivative
        ds = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ds[:, :-1] = numpy.diff(dataC, axis=-1) / numpy.diff(time)
        ds[:, -1] = 0

        return ts, ys.transpose(), ds.transpose()

    @staticmethod
    def bessel(data, **kwargs):
        """
        Implement a Bessel type analog filter
        ----------
        data: xarray DataSet as obtained from a TDI call

        Returns
        -------
        Smoothed array and time derivative
        """
        time = data.time.values
        dt = (time.max() - time.min()) / (time.size - 1)
        Ny = numpy.round(
            1. / ((time.max() - time.min()) / (time.size - 1))) / 2
        # implement an appropriate Bessel analog filter
        fcutoff = kwargs.get('fcutoff', 30.)
        _Wn = fcutoff / Ny
        b, a = signal.bessel(4, _Wn)
        # create a copy of the signals
        sm = data.values.transpose().copy()
        smd = signal.filtfilt(b, a, numpy.gradient(sm, dt, axis=-1), axis=-1)
        return sm.transpose(), smd.transpose()
