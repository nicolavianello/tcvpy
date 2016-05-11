"""

Fast Reciprocating Probe data and profiles

Written by Nicola Vianello

"""

import numpy as np
import tcv
import eqtools
import xray
import time

class FastRP(object):
    """

    Load the signals from the Fast Reciprocating Probe
    Diagnostic. It can load the data as a function of
    time or compute the appropriate profiles. The
    mapping from (R, Z) coordinates to rho is
    ensured by the eqtools class

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
        _string = 'getnci(getnci(\\TOP.DIAGZ.MEASUREMENTS.UCSDFP,'\
                  '"MEMBER_NIDS")'\
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
            iSat = conn.tdi(r'\FP'+_IsName[1])
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
                      trange=None, remote=False,
                      alltime=False):

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
        alltime: Boolean. Default False
            if set to true it compute the rho for each point
            in time and then average. Otherwise rho is computed
            at reduced timing (downsampled to npoint in time)
        Return:
        -------
        data: xray DataArray with rho poloidal as dimension and
        error as attributes. It also saves the absolute
        position as attribute
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

        if alltime:
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
        else:
            print 'downsampling in time'
            R = FastRP._getpostime(shot, stroke=stroke,
                                   remote=remote)
            if trange is None:
                trange = [R.time.min(), R.time.max()]
                iN = iSat.where(((iSat.time >= R.time.min()) &
                                 (iSat.time <= R.time.max()))).values
                iN = iN[~np.isnan(iN)]
                rN = R.values
                tN = R.time.values
            else:
                print 'limiting in time'
                trange = np.atleast_1d(trange)
                tN = R.time(((R.time >= trange[0]) &
                             (R.time <= trange[1]))).values
                rN = R.where(((R.time >= trange[0]) &
                              (R.time <= trange[1]))).values
                rN = rN[~np.isnan(rN)]
                iN = iSat.where(((iSat.time >= trange[0]) &
                                 (iSat.time <= trange[1]))).values
                iN = iN[~np.isnan(iN)]
            # slice also the timing
            out = FastRP._getprofileT(rN, iN, tN)
            # now convert into appropriate equilibrium
            rho = np.zeros(npoint)
            eq = eqtools.TCVLIUQETree(shot)
            for x, t, i in zip(out.R.values, out.time, range(npoint)):
                rho[i] = eq.rz2psinorm(x, 0, t, sqrt=True)
            data = xray.DataArray(out.values, coords={'rho': rho})
            data.attrs['err'] = out.err
            data.attrs['R'] = out.R.values

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
        _name = _name[_nameS == str(stroke)]
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
                      trange=None, remote=False,
                      alltime=False):

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
        alltime: Boolean. Default False
            if set to true it compute the rho for each point
            in time and then average. Otherwise rho is computed
            at reduced timing (downsampled to npoint in time)
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
        # probe associated to the radial position
        rDict = {}
        for p in vf.Probe.values:
            if p[2] == 'R':
                rDict[p] = 0.003
            else:
                rDict[p] = 0.

        if alltime:
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
                print 'Computing profile for ' + n
                y = vf.sel(Probe=n).values
                x = rho.sel(r=rDict[n]).values
                out = FastRP._getprofileR(x, y, npoint=npoint)
                dout[n] = out
        else:
            eq = eqtools.TCVLIUQETree(shot)
            R = FastRP._getpostime(shot, stroke=stroke,
                                   remote=remote, r=0.003)
            # now for each of the probe we perform the
            # evaluation of the profile
            if trange is None:
                trange = [R.time.min(), R.time.max()]
            # now limit all the signals to the appropriate timing
            vf = vf[:, ((vf.time >= trange[0]) &
                        (vf.time <= trange[1]))]
            R = R[:, ((R.time >= trange[0]) &
                      (R.time <= trange[1]))]
            dout = {}
            for n in vf.Probe.values:
                print 'Computing profile for ' + n
                y = vf.sel(Probe=n).values
                x = R.sel(r=rDict[n]).values
                t = R.time.values
                out = FastRP._getprofileT(x, y, t,  npoint=npoint)
                rho = np.zeros(npoint)
                for x, t, i in zip(out.R.values, out.time, range(npoint)):
                    rho[i] = eq.rz2psinorm(x, 0, t, sqrt=True)
                out2 = xray.DataArray(out.values, coords={'rho': rho})
                out2.attrs['err'] = out.err
                out2.attrs['R'] = out.R.values
                dout[n] = out2
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
    def _getpostime(shot, stroke=1, r=None,
                    remote=False):
        """
        Given the shot and the stroke it load the
        appropriate position of the probe.
        It save it
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
           respect to the front tip.
           It is given in [m] with positive
           values meaning the tip is behind the front one
        Return:
        ----------
        rProbe:xarray DataArray
            rProbe if given, the relative distance
            from the top tip. Remeber that it limits
            the time to the maximum position to the
            maximum available point in the psi grid
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> rProbe = FastRP._getpostime(51080, stroke=1, r=[0.001, 0.002])
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
        # rN should be in [m]
        rN = rN.values[~np.isnan(rN.values)]/1e2
        # ok now distinguish between the different cases
        #
        if r is not None:
            rOut = np.vstack((rN, rN+r))
            data = xray.DataArray(rOut, coords=[np.append(0, r), tN],
                                  dims=['r', 'time'])
        else:
            data = xray.DataArray(rN, coords={'time': tN})

        conn.close
        return data

    @staticmethod
    def _getprofileR(x, y, npoint=20):
        """
        Given x and y compute the profile assuming x is
        the coordinate and y the variable. It does it
        by sorting along x, splitting into npoint
        and computing the mean and standard deviation

        """
        y = y[np.argsort(x)]
        x = x[np.argsort(x)]
        yS = np.array_split(y, npoint)
        xS = np.array_split(x, npoint)
        yO = np.asarray([np.nanmean(k) for k in yS])
        xO = np.asarray([np.nanmean(k) for k in xS])
        eO = np.asarray([np.nanstd(k) for k in yS])
        data = xray.DataArray(yO, coords={'rho': xO})
        data.attrs['err'] = eO
        return data

    @staticmethod
    def _getprofileT(x, y, t, npoint=20):
        """
        Given x and y and corresponding time
        compute the profile assuming x is
        the coordinate and y the variable. It does it
        by splitting along the time dimension and
        save the average timing
        """

        y = y[np.argsort(t)]
        x = x[np.argsort(t)]
        t = t[np.argsort(t)]
        yS = np.array_split(y, npoint)
        xS = np.array_split(x, npoint)
        tS = np.array_split(t, npoint)
        yO = np.asarray([np.nanmean(k) for k in yS])
        xO = np.asarray([np.nanmean(k) for k in xS])
        eO = np.asarray([np.nanstd(k) for k in yS])
        tO = np.asarray([np.nanmean(k) for k in tS])
        data = xray.DataArray(yO, coords={'R': xO})
        data.attrs['err'] = eO
        data.attrs['time'] = tO
        return data

    @staticmethod
    def iSstatfromshot(shot, stroke=1,
                       npoint=20, remote=False):
        """
        It compute the profile as a function
        of rho or radius of the statisical properties
        of ion saturation current, namely the auto-correlation
        the rms the flatness and the skewness by
        slicing the signal in the given number of point

        Parameters:
        ----------
        shot: int
            Shot Number
        stroke: int. Default 1
            Choose between the 1st or 2nd stroke
        remote: Boolean. Default False
            If set it connect to 'localhost:1600' supposing
            an ssh forwarding is taking place
        npoint: Number of point for the radial resolution

        Return:
        ----------
        Dictionary containing the following keys:
        tac : auto-correlation time
        rms : rms of the signal
        skew: skewness
        flat: flatness
        Each value of the dictionary is an Xarray dataset
        including as dimension the rho and as attribute the
        absolute R position
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> rho = FastRP.rhofromshot(51080, stroke=1, r=[0.001, 0.002])
        >>> matplotlib.pylab.plot(vf.time, rho.values)

        """

        return -1
