"""

Fast Reciprocating Probe data and profiles

Written by Nicola Vianello

"""

import numpy as np
import tcv
import eqtools
import xray

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
            Server = 'localhost:1600'
        else:
            Server = 'tcvdata.epfl.ch'

        conn = tcv.shot(shot, server=Server)
        _string = 'getnci(getnci(\\TOP.DIAGZ.MEASUREMENTS.UCSDFP,"MEMBER_NIDS")'\
                  ',"NODE_NAME")'
        nodeName = conn.tdi(_string)
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
        >>> iSat = FastRP.iSTimefromshot(51080, stroke=1)
        >>> matplotlib.pylab.plot(iSat.time, iSat.values)
        """
        # assume we are blind and found the name appropriately
        _name = FastRP._getNodeName(shot, remote=remote)
        _nameS = np.asarray([n[:-3] for n in _name])
        # these are all the names of the ion saturation current
        # signal
        _IsName = _name[np.where(_nameS == 'IS')]
        if remote:
            Server = 'localhost:1600'
        else:
            Server = 'tcvdata.epfl.ch'
        conn = tcv.shot(shot, server=Server)
        if stroke == 1:
            iSat = conn.tdi(r'\FP'+_IsName[0])
        else:
            iSat = conn.tdi(r'\FPis1_2')
        # eventually combine the two
        # rename with time as dimension
        iSat = iSat.rename({'dim_0': 'time'})
        # add in the attributes also the area
        iSat.area = conn.tdi(r'\AM4').values
        # detrend the costant in the first part of the
        # stroke
        iSat -= iSat.where(iSat.time < iSat.time.min() +
                           0.01).mean(dim='time')
        # close the connection
        conn.close
        return iSat

    @staticmethod
    def iSRhofromshot(shot, stroke=1, npoint=20,
                      trange=None, remote=False):

        """
        Return the ion saturation current profile
        as a function of normalized poloidal flux
        save in an xray DataStructure
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
        data: xray DataArray with rho poloidal as dimension and
        error as attributes
        Example:
        --------
        >>> data = iSRhofromshot(51080, stroke=1, npoint=20)
        >>> matplotlib.pylab.plot(data.rho, data.values, 'o')
        >>> matplotlib.pylab.errorbar(data.rho, data.values,
                                      yerr=data.err, fmt='none')
        """

        # get the appropriate iSat
        iSat = FastRP.iSTimefromshot(shot, stroke=stroke,
                                     remote=remote)
        # now collect the position
        if remote:
            Server = 'localhost:1600'
        else:
            Server = 'tcvdata.epfl.ch'
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
        iN = iSat.where(cPos.values/1e2 < rMax).values
        # these are the timing
        tN = cPos.where(cPos.values/1e2 < rMax).dim_0.values
        tN = tN[~np.isnan(rN.values)]
        # this is the ion saturation current in this interval
        iN = iN[~np.isnan(rN.values)]
        # rN should be in rho
        rN = rN.values[~np.isnan(rN.values)]/1e2
        # now translate rN in rho
        rho = np.zeros(rN.size)
        for r, t, i in zip(rN, tN, range(tN.size)):
            rho[i] = eq.rz2psinorm(r, 0, t, sqrt=True)
        # now in case we also set a min-max of time we limit to this interval
        if trange is not None:
            trange = np.atleast_1d(trange)
            _idx = ((tN >= trange[0]) & (tN <= trange[1]))
            rho = rho[_idx]
            iN = iN[_idx]
            tN = tN[_idx]
        # now sort along rN and average corrispondingly
        iN = iN[np.argsort(rho)]
        rN = rho[np.argsort(rho)]
        iSliced = np.array_split(iN, npoint)
        rSliced = np.array_split(rN, npoint)
        iOut = np.asarray([np.nanmean(k) for k in iSliced])
        iErr = np.asarray([np.nanstd(k) for k in iSliced])
        rOut = np.asarray([np.nanmean(k) for k in rSliced])
        conn.close
        data = xray.DataArray(iOut, coords=[('rho', rOut)])
        data.attrs['err'] = iErr
        data.attrs['units'] = 'A'
        return data

    @staticmethod
    def VfTimefromshot(shot, stroke=1, remote=False):
        """
        Return the value of Floating potential as
        multi dimensional array. It is build in order
        to retain the name of the probe in the array in the
        coordinates
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
        vf: xarray Dataarray
            Return the floating potential from all
            the available probes. The data array has in
            coords the name of the signals
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> vf = FastRP.VfTimefromshot(51080, stroke=1)
        >>> matplotlib.pylab.plot(vf.time, vf.sel(Probe='VFT_1'))
        """
        _name = FastRP._getNodeName(shot, remote=remote)
        # first of all choose only those pertaining to the chosen stroke
        _nameS = np.asarray([n[-1] for n in _name])
        _name = _name[_nameS == '1']
        # now choose thich collect only vf
        _nameS = np.asarray([n[:2] for n in _name])
        _nameVf = _name[_nameS == 'VF']
        if remote:
            Server = 'localhost:1600'
        else:
            Server = 'tcvdata.epfl.ch'
        values = []
        names = []
        with tcv.shot(shot, server=Server) as conn:
            for s in _nameVf:
                values.append(conn.tdi(r'\fp'+s, dims='time'))
                names.append(s)
        data = xray.concat(values, dim='Probe')
        data['Probe'] = names
        # detrend initial part
        data -= data.where(data.time < data.time.min()+0.01).mean(dim='time')
        conn.close
        return data

    @staticmethod
    def VfRhofromshot(shot, stroke=1, npoint=20,
                      trange=None, remote=False):

        """
        Return the floating potential profile
        as a function of normalized poloidal flux for
        all the available probe array considering their
        position within the probe array.
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
        vf: ion saturation current 2D array
        err: rms of the signal (considering the space resolution)

        Example:
        --------
        >>> rho, iSat, err = iSRhofromshot(51080, stroke=1, npoint=20)
        >>> matplotlib.pylab.plot(rho, iSat, 'o')
        >>> matplotlib.pylab.errorbar(rho, iSat, yerr=err, fmt='none')
        """
        # first of all collect the floating potential
        vf = FastRP.VfTimefromshot(shot, stroke=stroke,
                                   remote=remote)
        # now create a dictionary which associate the
        # relative position of the probe tip with respect
        # to the calculated position of the front tip

    @staticmethod
    def rhofromshot(shot, stroke=1, r=None,
                    remote=False):
        """
        It compute the rho poloidal
        for the given stroke. It save it
        as a xray data structure with coords ('time', 'r')
        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        remote: Boolean. Default False
            If set it connect to 'localhost:1600' supposing
            an ssh forwarding is taking place
        r: Float
           This can be a floating or an ndarray containing
           the relative radial distance with respect to the
           front tip (useful to determine the rho corresponding)
           to back probes. It is given in [m] with positive
           values meaning the tip is behind the front one
        Return:
        ----------
        rho:xarray DataArray
            Rho, (sqrt(normalized flux)) with dimension time
            and second dimension if given the relative distance
            from the top tip
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> rho = FastRP.rhofromshot(51080, stroke=1)
        >>> matplotlib.pylab.plot(vf.time, rho.values)

        """
        # determine first of all the equilibrium
        eq = eqtools.TCVLIUQETree(shot)
        rMax = eq.getRGrid().max()
        if remote:
            Server = 'localhost:1600'
        else:
            Server = 'tcvdata.epfl.ch'
        # open the connection
        conn = tcv.shot(shot, server=Server)
        # now determine the radial location
        if stroke == 1:
            cPos = conn.tdi(r'\fpcalpos_1')
        else:
            cPos = conn.tdi(r'\fpcalpos_2')
        # limit our self to the region where equilibrium
        # is computed
        rN = cPos.where(cPos/1e2 < rMax)
        tN = cPos.where(cPos.values/1e2 < rMax).dim_0.values
        tN = tN[~np.isnan(rN.values)]
        # rN should be in rho
        rN = rN.values[~np.isnan(rN.values)]/1e2
        # ok now distinguish between the different cases
        #
        if r is not None:
            r = np.atleast_1d(r)
            rho = np.zeros((rN.size, r.size+1))
            for R, t, i in zip(rN, tN, range(tN.size)):
                rho[i, :] = eq.rz2psinorm(np.append(R, R+r),
                                          np.repeat(0, r.size+1),
                                          t, sqrt=True)

            data = xray.DataArray(rho, coords=[('time', tN), ('r', r)]),
        else:
            rho = np.zeros(rN.size)
            for R, t, i in zip(rN, tN, range(tN.size)):
                rho[i] = eq.rz2psinorm(R, 0, t, sqrt=True)
            data = xray.DataArray(rho, coords=[('time', tN)])

        return data
