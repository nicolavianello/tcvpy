"""
Fast Reciprocating Probe

Written by Nicola Vianello
"""

import os
import numpy as np
import xray
import tcv
import eqtools

class FastRP(object):
    """

    Load the signals from the Fast Reciprocating Probe
    Diagnostic. 

    """
    
    @classmethod
    def _getNodeName(shot):
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
        conn = tcv.shot(shot)
        _string = 'getnci(getnci(\\TOP.DIAGZ.MEASUREMENTS.UCSDFP,"MEMBER_NIDS")'\
                  ',"NODE_NAME")'
        nodeName = conn.tdi(_string)
        # transform in the 
        nodeName = np.core.defchararray.strip(nodeName.values)
        # close the connection
        conn.close
        return nodeName
    
    @staticmethod
    def iSTimefromshot(shot, stroke=1):
        """
        Return the ion saturation current from shot
        with the proper time base
        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        Return:
        ----------
        iSat: xarray Dataarray
            Return the collected ion saturation current
            including its time and area
        """
        # assume we are blind and found the name appropriately
        _name = FastRP._getNodeName(shot)
        _nameS = np.asarray([n[:-2] for n in _name])
        # these are all the names of the ion saturation current
        # signal
        _IsName = _name[np.where(_nameS == 'IS')]
        conn = tcv.shot(shot)
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

    @statimethod
    def iSRhofromshot(shot, stroke=1, npoint=20):

        """
        Return the ion saturation current from shot
        with the proper time base
        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        npoint: int. Default 20
            Number of point in space 
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
                                     npoint=npoint)
        # open the connection
        conn = tcv.shot(shot)
        # now determine the radial location
        if stroke ==1:
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
        iN = iN.val
