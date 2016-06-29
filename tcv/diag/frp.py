"""

Fast Reciprocating Probe data and profiles

Written by Nicola Vianello

"""

import numpy as np
import tcv
import eqtools
import xray
from scipy import stats
from scipy.interpolate import UnivariateSpline


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
            print 'Retrieveing data for stroke 1'
        else:
            iSat = conn.tdi(r'\FP'+_IsName[1])
            print 'Retrieveing data for stroke 2'
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
        # unfortunately we need the delta t is not
        # constant. We redefine the time basis
        _dTime = np.arange(iSat.size, dtype='double')*4.e-7 + \
                 iSat.time.min().item()
        iSat.time.values = _dTime

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
        xRay DataArray with coordinates the rho and as
        attributes the error, the absolute radial position
        and the R-Rsep position, upstream remapped. As attributes we
        also save the UnivariateSpline object for the profile along
        the 3 saved x (rho, R, R-Rsep Upstream Remapped)

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
            # get the appropriate R-Rsep
            rresp = FastRP.Rrsepfromshot(shot, stroke=stroke,
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
            # save into the data also the object for spline interpolation
            data.attrs['spline'] = UnivariateSpline(rOut,
                                                    iOut,
                                                    w=1./iErr, ext=0)
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
                tN = R.time[((R.time >= trange[0]) &
                             (R.time <= trange[1]))].values
                rN = R.where(((R.time >= trange[0]) &
                              (R.time <= trange[1]))).values
                rN = rN[~np.isnan(rN)]
                iN = iSat.where(((iSat.time >= trange[0]) &
                                 (iSat.time <= trange[1]))).values
                iN = iN[~np.isnan(iN)]
            # slice also the timing
            out = FastRP._getprofileT(rN, iN, tN,
                                      npoint=npoint)
            # now convert into appropriate equilibrium
            rho = np.zeros(npoint)
            rrsep = np.zeros(npoint)
            eq = eqtools.TCVLIUQETree(shot)
            for x, t, i in zip(out.R.values, out.time, range(npoint)):
                rho[i] = eq.rz2psinorm(x, 0, t, sqrt=True)
                rrsep[i] = eq.rz2rmid(x, 0, t)- eq.getRMidOutSpline()(t)
            _id = np.argsort(rho)
            data = xray.DataArray(out.values[_id],
                                  coords={'rho': rho[_id]})
            data.attrs['err'] = out.err[_id]
            data.attrs['R'] = out.R.values[_id]
            data.attrs['Rrsep'] = rrep[_id]
            data.attrs['Rsp'] = UnivariateSpline(data.attrs['R'],
                                                 data.values,
                                                 w=1./data.err, ext=0)
            data.attrs['Rhosp'] = UnivariateSpline(data.rho.values,
                                                   data.values,
                                                   w=1./data.err, ext=0)
            data.attrs['RrsepSp'] = UnivariateSpline(data.attrs['Rrsep'],
                                                     data.values, w=1./data.err
                                                     ext=0)
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
        # build the timing in double precision
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
        Dictionaro of xRay DataArray with coordinates the rho and as
        attributes the error, the absolute radial position
        and the R-Rsep position, upstream remapped
        
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
                out = FastRP._getprofileT(x, y, t,
                                          npoint=npoint)
                rho = np.zeros(npoint)
                for x, t, i in zip(out.R.values, out.time, range(npoint)):
                    rho[i] = eq.rz2psinorm(x, 0, t, sqrt=True)
                _id = np.argsort(rho)
                out2 = xray.DataArray(out.values[_id],
                                      coords={'rho': rho[_id]})
                out2.attrs['err'] = out.err[_id]
                out2.attrs['R'] = out.R.values[_id]
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
        rN = FastRP._getpostime(shot, stroke=stroke, remote=remote,
                                r=r)
        if r is not None:
            r = np.atleast_1d(r)
            rho = np.zeros((r.size+1, rN.shape[1]))
            for R, t, i in zip(rN.values[0, :],
                               rN.time.values,
                               range(rN.time.size)):
                rho[:, i] = eq.rz2psinorm(np.append(R, R+r),
                                          np.repeat(0, r.size+1),
                                          t, sqrt=True)

            data = xray.DataArray(rho, coords=[np.append(0, r),
                                               rN.time.values],
                                  dims=['r', 'time'])
        else:
            rho = np.zeros(rN.size)
            for R, t, i in zip(rN.values, rN.time.values,
                               range(rN.time.size)):
                rho[i] = eq.rz2psinorm(R, 0, t, sqrt=True)
            data = xray.DataArray(rho, coords={'time': rN.time.values})

        return data

    @staticmethod
    def Rrsepfromshot(shot, stroke=1, r=None,
                    remote=False):
        """
        It compute the R-Rsep array as a function of time
        for a given stroke. It save it
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
        Rrsep:xarray DataArray
            R-Rsep upstream remapped with dimension time
        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> Rrsep = FastRP.Rrsepfromshot(51080, stroke=1, r=[0.001, 0.002])

        """
        # determine first of all the equilibrium
        eq = eqtools.TCVLIUQETree(shot)
        rN = FastRP._getpostime(shot, stroke=stroke, remote=remote,
                                r=r)
        if r is not None:
            r = np.atleast_1d(r)
            rho = np.zeros((r.size+1, rN.shape[1]))
            for R, t, i in zip(rN.values[0, :],
                               rN.time.values,
                               range(rN.time.size)):
                rho[:, i] = eq.rz2rmid(np.append(R, R+r),
                                          np.repeat(0, r.size+1),
                                          t) - eq.getRmidOutSpline()(t)

            data = xray.DataArray(rho, coords=[np.append(0, r),
                                               rN.time.values],
                                  dims=['r', 'time'])
        else:
            rho = np.zeros(rN.size)
            for R, t, i in zip(rN.values, rN.time.values,
                               range(rN.time.size)):
                rho[i] = eq.rz2rmid(R, 0, t) - \
                         eq.getRmidOutSpline()(t)
            data = xray.DataArray(rho, coords={'time': rN.time.values})

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

        # we need to find the minimum maximum time where
        # the probe is within the range of psiRZ
        trange = [cPos[(cPos/1e2 + add) < rMax].dim_0.values.min(),
                  cPos[(cPos/1e2 + add) < rMax].dim_0.values.max()]
        rN = cPos[((cPos.dim_0 > trange[0]) &
                   (cPos.dim_0 < trange[1]))].values/1e2
        tN = cPos[((cPos.dim_0 > trange[0]) &
                   (cPos.dim_0 < trange[1]))].dim_0.values
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
                       npoint=20, trange=None,
                       remote=False):
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
        xarray DataArray with dimension rho and values the
        auto-correlation time. As attribute it saves also
        rms : rms of the signal
        skew: skewness
        flat: flatness

        Example:
        --------
        >>> from tcv.diag.frp import FastRP
        >>> stat = FastRP.iSstatfromshot(51080, stroke=1)
        >>> matplotlib.pylab.plot(stat.rho, stat.values)

        """

        # the the position in R
        R = FastRP._getpostime(shot, stroke=stroke, remote=remote)
        # the the iSat
        iS = FastRP.iSTimefromshot(shot, stroke=stroke, remote=remote)
        #
        if trange is None:
            trange = [R.time.min().item(), R.time.max().item()]
        # limit to the same timing
        iS = iS[((iS.time >= trange[0]) &
                 (iS.time <= trange[1]))].values
        T = R[((R.time >= trange[0]) &
               (R.time <= trange[1]))].time.values
        R = R[((R.time >= trange[0]) &
               (R.time <= trange[1]))].values
        dt = (T.max()-T.min())/(T.size-1)
        # now we slice appropriately
        iSS = np.array_split(iS, npoint)
        tau = np.zeros(npoint)
        rms = np.asarray([k.std() for k in iSS])
        Fl = np.asarray([stats.kurtosis(k, fisher=False) for k in iSS])
        Sk = np.asarray([stats.skew(k) for k in iSS])
        for i in range(np.size(iSS)):
            dummy = iSS[i]
            c = np.correlate(dummy, dummy, mode='full')
            # normalize
            c /= c.max()
            lag = np.arange(c.size)-c.size/2
            # check if c.min > c/2 then is NaN
            if (c.min() >= 1./np.exp(1)):
                tau[i] = np.float('nan')
            else:
                tau[i] = 2*np.abs(lag[np.argmin(np.abs(c-1/np.exp(1)))])*dt
        # determine the position
        rO = np.asarray([np.nanmean(k) for k
                         in np.array_split(R, npoint)])
        tO = np.asarray([np.nanmean(k) for k
                         in np.array_split(T, npoint)])
        # convert in rho
        eq = eqtools.TCVLIUQETree(shot)
        rho = np.zeros(npoint)
        for r, t, i in zip(rO, tO, range(npoint)):
            rho[i] = eq.rz2psinorm(r, 0, t, sqrt=True)
        data = xray.DataArray(tau, coords={'rho': rho})
        data.attrs['rms'] = rms
        data.attrs['Flat'] = Fl
        data.attrs['Skew'] = Sk
        return data

    @staticmethod
    def _hastroke(shot, stroke=1):
        """
        Simple method which return a boolean according
        to the fact the stroke has worked or not
        """
        try:
            iS = FastRP.iSTimefromshot(shot, stroke=stroke)
            if (iS.max()-iS.min()) > 0.1:
                return True
            else:
                return False
        except:
            return False
