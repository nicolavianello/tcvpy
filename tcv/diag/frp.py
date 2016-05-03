"""

Fast Reciprocating Probe data and profiles

Written by Nicola Vianello

"""

import numpy as np
import tcv
import eqtools

class FastRP(object):
    """

    Load the signals from the Fast Reciprocating Probe
    Diagnostic. 

    """
    
    @staticmethod
    def _getNodeName(shot, remote=False):
        """
        Class method to obtain the name of the node in the 
        desidered shot number.

        Parameters:
        ----------
        shot: int
            Shot Number
        remote: Boolean. Default False
            If set it connect to 'localhost:1600' supposing
            an ssh forwarding is taking place
        Return:
        ----------
        nodeName = String array
            String containing the name of the signal
            saved into the node.

        """
        if remote:
            Server='localhost:1600'
        else:
            Server='tcvdata.epfl.ch'

        conn = tcv.shot(shot, server=Server)
        _string = 'getnci(getnci(\\TOP.DIAGZ.MEASUREMENTS.UCSDFP,"MEMBER_NIDS")'\
                  ',"NODE_NAME")'
        nodeName = conn.tdi(_string)
        # transform in the 
        nodeName = np.core.defchararray.strip(nodeName.values)
        # close the connection
        conn.close
        return nodeName
    
    @staticmethod
    def iSTimefromshot(shot, stroke=1, remote=False):
        """
        Return the ion saturation current from shot
        with the proper time base
        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        remote: Boolean. Default False
            If set it connect to 'localhost:1600' supposing
            an ssh forwarding is taking place
        Return:
        ----------
        iSat: xarray Dataarray
            Return the collected ion saturation current
            including its time and area
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> iSat = FastRP.iSfromshot(51080, stroke=1)
        >>> matplotlib.pylab.plot(iSat.time, iSat.values)
        """
        # assume we are blind and found the name appropriately
        _name = FastRP._getNodeName(shot, remote=remote)
        _nameS = np.asarray([n[:-3] for n in _name])
        # these are all the names of the ion saturation current
        # signal
        _IsName = _name[np.where(_nameS == 'IS')]
        if remote:
            Server='localhost:1600'
        else:
            Server='tcvdata.epfl.ch'
        conn = tcv.shot(shot, server=Server)
        if stroke == 1:
            iSat = conn.tdi(r'\FP'+_IsName[0])
        else:
            iSat = conn.tdi(r'\FPis1_2')
        # eventually combine the two
        # rename with time as dimension
        iSat=iSat.rename({'dim_0':'time'})
        # add in the attributes also the area
        iSat.area = conn.tdi(r'\AM4').values
        # close the connection
        conn.close
        return iSat

    @staticmethod
    def iSRhofromshot(shot, stroke=1, npoint=20,
                      trange=None, remote=False):

        """
        Return the ion saturation current profile
        as a function of normalized poloidal flux
        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        npoint: int. Default 20
            Number of point in space 
        trange: 2D array or list
            eventually we can use a limited
            number of point (for example just entering
            the plasma). Default it uses all the stroke
        remote: Boolean. Default False
            If set it connect to 'localhost:1600' supposing
            an ssh forwarding is taking place
        Return:
        -------
        rho: Normalized poloidal flux coordinate
        iSat: ion saturation current
        err: rms of the signal (considering the space resolution)

        Example:
        --------
        >>> rho, iSat, err = iSRhofromshot(51080, stroke=1, npoint=20)
        >>> matplotlib.pylab.plot(rho, iSat, 'o')
        >>> matplotlib.pylab.errorbar(rho, iSat, yerr=err, fmt='none')
        """

        # get the appropriate iSat
        iSat = FastRP.iSTimefromshot(shot, stroke=stroke,
                                     remote=remote)
        # now collect the position
        if remote:
            Server='localhost:1600'
        else:
            Server='tcvdata.epfl.ch'
        # open the connection
        conn = tcv.shot(shot, server=Server)
        # now determine the radial location
        if stroke == 1:
            cPos = conn.tdi(r'\fpcalpos_1')
        else:
            cPos = conn.tdi(r'\fpcalpos_2')
        # now determine the time and space where the
        # position of the probe is within
        eq = eqtools.TCVLIUQETree(shot)
        rMax = eq.getRGrid().max()
        # this is the region where it is
        # within the grid of the flux the position
        rN = cPos.where(cPos/1e2 < rMax)
        iN = iSat.where(cPos/1e2 < rMax).values
        # these are the timing
        tN = cPos.where(cPos/1e2 < rMax).dim_0.values
        tN = tN[~np.isnan(rN.values)]
        # this is the ion saturation current in this interval
        iN = iN[~np.isnan(rN.Values)]
        rN = rN.values[~np.isnan(rN.values)]
        # now in case we also set a min-max of time we limit to this interval
        if trange is not None:
            trange = np.atleast_1d(trange)
            _idx = ((tN>=trange[0]) & (tN <= trange[1]))
            rN = rN[_idx]
            iN = iN[_idx]
            tN = tN[_idx]
        # now sort along rN and average corrispondingly
        iN = iN[np.argsort(rN)]
        rN = rN[np.argsort(rN)]
        iSliced = np.array_split(iN, npoint)
        rSliced = np.array_splint(rN, npoint)
        iOut = np.asarray([np.nanmean(k) for k in iSliced])
        iErr = np.asarray([np.nanstd(k) for k in iSliced])
        rOut = np.asarray([np.nanmean(k) for k in rSliced])
        return rOut, iOut, iErr
