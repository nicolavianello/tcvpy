# -*- coding: utf-8 -*-
"""
XTOMO diagnostics data

Written by Nicola Vianello
"""

import logging
import os  # for correctly handling the directory position
from scipy import io  # this is needed to load the settings of xtomo
import numpy as np

import tcv  # this is the tcv main library component
import xarray as xray  # this is needed as tdi save into an xray


log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Top(object):
    """
    Class to load and analyze the data contained in the top chamber of DMPX.
    All the methods described as classmethod and so they can be called directly
    Methods:
        read: Read the data calibrated in the same way as mpxdata matlab
              routine.  It can read also single chords, accordingly to the call
              to Top
        gains: Return the gains used for the calibration of the chors
        geo : return the geometrical position of the chords
        spectrogram: It create and eventually plot the spectrograms(s)
    """

    @staticmethod
    def channels(shot, los=None):
        """
        Parameters
        ----------
        Same as fromshot.

        Returns
        -------
        String array containing the channels, already sorted out according to
        shot number
        """

        if los is None:
            los = np.arange(64) + 1
        else:
            los = np.atleast_1d(los)

        # we provide here all the definition we need
        if shot < 24087 or shot > 24725:
            dtNe1 = r'\atlas::dt100_northeast_001:'
            dtNe2 = r'\atlas::dt100_northeast_002:'
        else:
            dtNe1 = r'\atlas::dt100_southwest_001:'
            dtNe2 = r'\atlas::dt100_southwest_002:'

        # we define the table corresponding to correct cabling of the signals
        # cabling for the shots > 26555
        if shot >= 26555:
            cd2Cords = np.arange(1, 65, 2)
            cd2Chans = np.arange(1, 33, 1)
            cd1Cords = np.arange(2, 66, 2)
            cd1Chans = np.append(np.arange(16, 0, -1), np.arange(32, 16, -1))
        elif shot > 20643 and shot < 26555:
            cd2Cords = np.arange(1, 65, 2)
            cd2Chans = np.arange(1, 33, 1)
            cd1Cords = np.arange(2, 66, 2)
            cd1Chans = np.arange(32, 0, -1)
        else:
            raise Exception('This programs does not load shots before 20643')

        # now we define the appropriate board according to the shot number
        # and the presence of fast signals
        cd2Cards = np.repeat(dtNe2, cd2Cords.size)
        cd1Cards = np.repeat(dtNe1, cd1Cords.size)
        # now we collect all together the signals and sort from HFS to LFS
        allCords = np.append(cd1Cords, cd2Cords)
        allChans = np.append(cd1Chans, cd2Chans)
        allCards = np.append(cd1Cards, cd2Cards)
        # sort the index from HFS to LFS
        allChans = allChans[np.argsort(allCords)]
        allCards = allCards[np.argsort(allCords)]

        # now select the chosen diods
        indexLos = los - 1  # we convert from LoS to index
        _chosenChans = allChans[indexLos]
        _chosenCards = allCards[indexLos]

        return _chosenCards, _chosenChans

    @staticmethod
    def fromshot(shot, los=None):
        """
        Return the calibrated DMPX signals.

        Parameters
        ----------
        shot : int or MDSConnection
            Shot number or connection instance
        camera : int
            Number of the XTOMO camera
        los : int or sequence of ints
            Optional argument with lines of sight (LoS) of the chosen camera.
            If None, it loads all the 20 channels

        Returns
        -------

        An xray.DataArray containing the data, time basis and all the
        information in dictionary

        Examples
        --------
        >>> from tcv.diag.dmpx import Top
        >>> data = Top.fromshot(50730, los=32)
        """

        if los is None:
            los = np.arange(64) + 1
        else:
            los = np.atleast_1d(los)

        cards, channels = Top.channels(shot, los=los)
        Top._check_dtaq_trigger(shot)
        fast = Top._is_fast(shot, cards[0], channels[0])

        values = []
        with tcv.shot(shot) as conn:
            for card, channel in zip(cards, channels):
                values.append(conn.tdi(Top._node(card, channel, fast),
                              dims='time'))

        # now we create the xray
        data = xray.concat(values, dim='los')
        data['los'] = los
        # correct for bad timing
        if shot > 19923 and shot < 20939:
            data.time = (data.time + 0.04) * 0.9697 - 0.04
        # correct for missing shots
        if shot >= 25102 and shot < 25933 and (36 in los):
            data.value[np.argmin(np.abs(los - 36)), :] = 0
            log.warn('Channel 36 was missing for this shot')
        elif shot >= 27127 and shot <= 28124:
            # missing channels, problem with cable 2, repaired by DF in Dec
            # 2004
            missing = np.asarray([3, 64, 62, 60, 58, 56])
            for fault in missing:
                if fault in los:
                    data.values[np.argmin(np.abs(los - fault)), :] = 0
                    log.warn('Channel %s missing for this shot', fault)

            if shot >= 27185 and (44 in los):  # one more channel missing !...
                data.values[np.argmin(np.abs(los-44)), :] = 0
                log.warn('Channel 44 missing for this shot')

        if shot >= 28219 and shot < 31446:
            missing = np.asarray([19, 21])
            for fault in missing:
                if fault in los:
                    data.values[np.argmin(np.abs(los-fault)), :] = 0
                    log.warn('Channel %s missing for this shot', fault)
        # chose calibrate the signals
        # read the gain
        _calib, _gains = Top.gains(shot, los=los)
        data.values *= (_calib / _gains).reshape(los.size, 1)

        return data

    @staticmethod
    def gains(shot, los=None):

        """
        Provide the proper calibration factor for the signals so that can
        obtain directly the calibration used call:

            Calibration, Gain = Top.gains()

        """

        conn = tcv.shot(shot)

        # After #26575, the real voltage is indicated in the Vista window,
        # before it was the reference voltage
        mm = 500 if shot < 26765 else 1
        voltage = conn.tdi(
            r'\VSYSTEM::TCV_PUBLICDB_R["ILOT:DAC_V_01"]').values * mm
        if shot == 32035:
            voltage = 2000

        # we first load all the calibration and then choose the correct one
        # according to the ordering
        gainC = np.zeros(64)
        gainR = np.zeros(64)

        I = np.where(shot >= np.r_[20030, 23323, 26555, 27127, 29921, 30759,
                                   31446])[0][-1]
        if I == 0:
            log.warn('Detector gain dependence on the high voltage value not '
                     'included in the signal calibration')
            calib = Top.calibration_data('mpx_calib_first.mat')
            calib_coeff_t = np.squeeze(calib['calib_coeff'])
            gainC[:] = 1
        if I == 1:
            log.warn('Detector gain dependence on the high voltage value not '
                     'included in the signal calibration')
            calib = Top.calibration_data('mpx_calib_sept02.mat')
            calib_coeff_t = np.squeeze(calib['calib_coeff'])
            gainC[:] = 1
        if I == 2:
            log.warn('Detector gain dependence on the high voltage value not '
                     'included in the signal calibration')
            log.warn('There were leaks in the top detector wire chamber for '
                     '26554<shot<27128')
            log.warn('Calibration is not very meaningful')
            calib = Top.calibration_data('mpx_calib_may04.mat')
            calib_coeff_t = np.np.squeeze(calib['calib_coeff'])
            gainC[:] = 1
        if I == 3:
            log.warn(
                'Same gain dependence on the high voltage value taken for '
                'each channel')
            calib = Top.calibration_data('mpx_calib_july04.mat')
            calib_coeff = np.squeeze(calib['calib_coeff'])
            R = np.squeeze(calib['R'])
            calib_coeff_t = calib_coeff
            calib = Top.calibration_data('mpx_calib_may05.mat')
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            C = np.mean(C[:, :64], 1)  # Use the same gain for each channel
            gainC[:] = np.exp(np.interp(voltage, V, np.log(C)))
            gainR[:] = R
        if I == 4:
            log.warn(
                'Same gain dependence on the high voltage value taken for '
                'each channel')
            calib = Top.calibration_data('mpx_calib_may05.mat')
            calib_coeff = np.np.squeeze(calib['calib_coeff'])
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            calib_coeff_t = calib_coeff
            C = np.mean(C[:, :64], 1)  # Use the same gain for each channel
            gainC[:] = np.exp(np.interp(voltage, V, np.log(C)))
            # use the previous relative calibration
            gainR[:] = Top.calibration_data('mpx_calib_july04.mat')['R']
        if I == 5:
            # In this case, the different behaviour of the wires is contained
            # in the matrix of gains.  The calibration coefficients are in a
            # vector: one value per wire, same value for all tensions.
            log.info(
                'Gain dependence on the high voltage value calibrated for '
                'each channel')
            log.warn(
                'Leaks in the bottom detector, no relative calibration of '
                'the two detectors')
            calib = Top.calibration_data('mpx_calib_oct05.mat')
            calib_coeff = np.squeeze(calib['calib_coeff'])
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            calib_coeff_t = calib_coeff
            # Interpolation to get the proper gains wrt to the high tension
            # value
            gainC[:] = [np.interp(voltage, V, np.log(C[:, jj]))
                        for jj in range(64)]
            gainR[:] = np.nan
        if I == 6:
            # In this case, the different behaviour of the wires is contained
            # in the matrix of calibration coefficients.  The gains are in a
            # vector: one value per tension, same value for all wires.
            log.info(
                'Gain dependence on the high voltage value calibrated for '
                'each channel')
            calib = Top.calibration_data('mpx_calib_dec05_bis.mat')
            calib_coeff_top = np.squeeze(calib['calib_coeff_top'])
            C_top_av = np.squeeze(calib['C_top_av'])
            V_top = np.squeeze(calib['V_top'])
            R = np.squeeze(calib['R'])
            calib_coeff_t = []
            for jj in range(64):
                # Interpolation to get the proper calibration coefficient wrt
                # the high tension value
                calib_coeff_t.append(
                    np.interp(voltage, V_top, calib_coeff_top[:, jj]))
            gainC[:] = np.exp(np.interp(voltage, V_top, np.log(C_top_av)))
            gainR = R
        # now we can order accordingly to the index for shot > 26555
        # the ordering of the calibration is awful

        II = np.zeros(64, dtype=int)
        if shot >= 26555:
            II[0:63:2] = np.r_[32:64]
            II[1:64:2] = np.r_[15:-1:-1, 31:15:-1]
            calib_coeff_t = np.asarray(calib_coeff_t)[II]
            gainC = gainC[II]

        # limit the output to the chosen diods
        if los is None:
            indexLos = np.arange(64)
        else:
            indexLos = np.atleast_1d(los) - 1

        cOut = calib_coeff_t[indexLos]
        gOut = gainC[indexLos]

        return cOut, gOut

    @staticmethod
    def geo(shot):
        """

        Parameters
        ----------
        Input:
            shot: shot number

        Returns
        -------
            The radial and vertical coordinates of the LoS

        Examples
        -------
        In [1]: from tcv.diag.dmpx import Top
        In [2]: x, y = Top.geo(50766)
        """

        indices = np.arange(64)

        _geoF = Top.calibration_data('dmpx_los.mat')
        xc = _geoF['xchordt'][:, indices]
        yc = _geoF['ychordt'][:, indices]

        return xc, yc

    @staticmethod
    def calibration_data(name):
        return io.loadmat(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'dmpxcalib', name))

    @staticmethod
    def _check_dtaq_trigger(shot):
        with tcv.shot(shot) as conn:
            if shot < 24087 or shot > 24725:
                dtNe1 = r'\atlas::dt100_northeast_001:'
                dtNe2 = r'\atlas::dt100_northeast_002:'
            else:
                dtNe1 = r'\atlas::dt100_southwest_001:'
                dtNe2 = r'\atlas::dt100_southwest_002:'

            mode1 = conn.tdi(dtNe1 + 'MODE').values
            mode2 = conn.tdi(dtNe2 + 'MODE').values
            mode3 = 4

            mode = mode1 * mode2 * mode3
            if conn.shot >= 24087 and mode != 64:
                log.warn(
                    'Random temporal gap (5 to 25 us) between the two or '
                    'three MPX acquisition cards.')
                log.warn(
                    'Random temporal gap (0.1 to 0.2ms) between DTACQ and '
                    'TCV.')
                raise Warning('DTACQ not in mode 4')

    @staticmethod
    def _is_fast(shot, card, channel):
        """ We check if fast nodes are collected for shot below 34988 and
        eventually load fast data.
        """

        if shot > 34988:
            log.info('Loading fast data after big opening')
            return False

        with tcv.shot(shot) as conn:
            try:
                conn.tdi('{}selected:channel_{:03}'.format(card, channel))
                log.info('Loading high frequency for old shot')
                return True
            except:
                log.info('Loading low frequency for old shot')
                return False

    @staticmethod
    def _node(card, channel, fast):
        if fast:
            return '{}selected:channel_{:03}'.format(card, channel)
        else:
            return '{}channel_{:03}'.format(card, channel)
