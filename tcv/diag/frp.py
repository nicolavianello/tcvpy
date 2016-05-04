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

        # get the appropriate rho
        rho = FastRP.rhofromshot(shot, stroke=stroke,
                                 remote=remote)
        # now we need to limit to the tMin and tMax
        # this is true both in case it is given or
        # in the case we have rho
        if trange is None:
            trange = [rho.time.min(), rho.time.max()]
            iN = iSat.where(((iSat.time >= rho.time.min()) &
                             (iSat.time <= rho.time.max()))).values
            iN = iN[~np.isnan(iN)]
            rN = rho.values
        else:
            print 'limiting in time'
            trange = np.atleast_1d(trange)
            rN = rho.where(((rho.time >= trange[0]) &
                            (rho.time <= trange[1]))).values
            rN = rN[~np.isnan(rN)]
            iN = iSat.where(((iSat.time >= trange[0]) &
                             (iSat.time <= trange[1]))).values
            iN = iN[~np.isnan(iN)]
        # now sort along rN and average corrispondingly
        iN = iN[np.argsort(rN)]
        rN = rN[np.argsort(rN)]
        iSliced = np.array_split(iN, npoint)
        rSliced = np.array_split(rN, npoint)
        iOut = np.asarray([np.nanmean(k) for k in iSliced])
        iErr = np.asarray([np.nanstd(k) for k in iSliced])
        rOut = np.asarray([np.nanmean(k) for k in rSliced])
        data = xray.DataArray(iOut, coords={'rho': rOut})
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
        A dictioary with keys equal to the name of the probe and value an
        xRay DataArray with coordinates the rho and attributes the error

        Example:
        --------
        >>> d = VfRhofromshot(51080, stroke=1, npoint=20)
        >>> matplotlib.pylab.plot(d['VFB_1'].rho, d['VFB_1'].values, 'o')
        """
        # first of all collect the floating potential
        vf = FastRP.VfTimefromshot(shot, stroke=stroke,
                                   remote=remote)
        # read the probe name and create a dictionary with
        # probe associated to the
        rDict = {}
        for p in vf.Probe.values:
            if p[3] == 'R':
                rDict[p] = 0.003
            else:
                rDict[p] = 0.

        # now create a dictionary which associate the
        # relative position of the probe tip with respect
        # to the calculated position of the front tip
        rho = FastRP.rhofromshot(shot, stroke=stroke,
                                 remote=remote, r=0.003)
        # now for each of the probe we perform the
        # evaluation of the profile
        if trange is None:
            trange = [rho.time.min(), rho.time.max()]
        # now limit all the signals to the appropriate timing
        vf = vf[:, ((vf.time >= trange[0]) &
                    (vf.time <= trange[1]))]
        rho = rho[:, ((rho.time >= trange[0]) &
                      (rho.time <= trange[1]))]

        # now we can use the groupby and apply method
        # of xray
        dout = {}
        for n in vf.Probe.values:
            y = vf.sel(Probe=n).values
            x = rho.sel(r=p[n]).values
            out = FastRP._getprofile(x, y, npoint=npoint)
            dout[n] = out
        return dout
    
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
           This can be a floating or an ndarray or a list
           containing the relative radial distance with
           respect to the front tip (useful to determine
           the rho corresponding to back probes).
           It is given in [m] with positive
           values meaning the tip is behind the front one
        Return:
        ----------
        rho:xarray DataArray
            Rho, (sqrt(normalized flux)) with dimension time
            and second dimension, if given, the relative distance
            from the top tip
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> rho = FastRP.rhofromshot(51080, stroke=1, r=[0.001, 0.002])
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
        # to avoid the problem of None values if we are
        # considering also retracted position we distingish
        if r is None:
            add = 0
        else:
            add = np.atleast_1d(r).max()

        rN = cPos.where((cPos/1e2+add) < rMax)
        tN = cPos.where((cPos.values/1e2+add) < rMax).dim_0.values
        tN = tN[~np.isnan(rN.values)]
        # rN should be in rho
        rN = rN.values[~np.isnan(rN.values)]/1e2
        # ok now distinguish between the different cases
        #
        if r is not None:
            r = np.atleast_1d(r)
            rho = np.zeros((r.size+1, rN.size))
            for R, t, i in zip(rN, tN, range(tN.size)):
                rho[:, i] = eq.rz2psinorm(np.append(R, R+r),
                                          np.repeat(0, r.size+1),
                                          t, sqrt=True)

            data = xray.DataArray(rho, coords=[np.append(0, r), tN],
                                  dims=['r', 'time'])
        else:
            rho = np.zeros(rN.size)
            for R, t, i in zip(rN, tN, range(tN.size)):
                rho[i] = eq.rz2psinorm(R, 0, t, sqrt=True)
            data = xray.DataArray(rho, coords={'time': tN})

        conn.close
        return data

    @staticmethod
    def _getprofile(x, y, npoint=20):
        """
        Given x and r compute the profile with the given
        number of point and 

        """
        y = y[np.argsort(x)]
        x = x[np.argsort(x)]
        yS = np.array_split(y, npoint)
        xS = np.array_split(x, npoint)
        yO = np.asarray([np.nanmean(k) for k in yS])
        xO = np.asarray([np.nanmean(k) for k in xS])
        eO = np.asarray([np.nanstd(k) for k in yS])
        data = xray.DataArray(yO, coords={'rho':xO})
        data.attrs['err'] = eO
        return data
