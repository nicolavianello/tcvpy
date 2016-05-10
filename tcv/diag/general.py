# CLASS containing the set of fromtree to access some general
# time traces like current, bt, elongation, triangularity,
# line-average density
# central density from thomson, central temperature from thomson,
# central temperature from
# double filter
import tcv
import matplotlib as mpl


class General(object):
    """
    Python class to access some general signals for TCV.
    Implemented method (as @classmethod) are
    ip : read plasma current
    bphi : read the magnetic field on axis
    neline: Line integrated density
    neavg : Line average Density
    betapol: poloidal beta
    q: q. Can load q0 or q95 depending on call
    delta: Triangularity. Both at the edge and at 95% of the flux
    kappa: Elongation. Both at the edce and at 95% of the flux
    te: central temperature from Double Filter technique
    """

    @staticmethod
    def ip(shot, plt=False):
        """
        Load the plasma current
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces

        Returns
        -------
        xray Data set with the current time traces

        Examples
        -------
        >>> from tcv.diag import General
        >>> ip = General.ip(50882, plt=True)
        """

        data = tcv.shot(shot).tdi('tcv_ip()')
        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values / 1e3)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :] / 1e3)
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(r'I$_p$ [kA]')
        data = data.rename({'dim_0': 'time'})
        return data

    @staticmethod
    def bphi(shot, plt=False):
        """
        Load the Toroidal field
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces

        Returns
        -------
        xray Data set with the toroidal field time traces

        Examples
        -------
        >>> from tcv.diag import General
        >>> bphi = General.bphi(50882,plt=True)
        """

        data = tcv.shot(shot).tdi('tcv_bphi()')
        data.values *= -1
        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :])
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(r'B$_{\phi}$ [T]')

        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def neline(shot, plt=False):
        """
        Load the Line Integrated Density
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces

        Returns
        -------
        xray Data set with the Line Integrated Density
        field time traces

        Examples
        -------
        >>> from tcv.diag import General
        >>> neLine = General.neLine(50882,plt=True)
        """

        data = tcv.shot(shot).tdi(r'\results::fir:lin_int_dens')
        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values / 1e19)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :] / 1e19)
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(r'n$_{e}$ Line Integrated [10$^{19}$ fringes]')
        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def neavg(shot, plt=False):
        """
        Load the Line Average Density
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces

        Returns
        -------
        xray Data set with the Line Average Density field time traces

        Examples
        -------
        >>> from tcv.diag import General
        >>> neAvg = General.neAvg(50882,plt=True)
        """

        data = tcv.shot(shot).tdi(r'\results::fir:n_average')
        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values / 1e19)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :] / 1e19)
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(r'n$_{e}$ Line Average [10$^{19}$ m$^{-3}$]')
        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def q(shot, plt=False, edge=False):
        """
        Load q (q95 or qedge) time trace
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces
        edge: Boolean (default is False). If it is True it
              loads the qedge rather than the q95
        Returns
        -------
        xray Data set with the q computed as default at 95% of
        poloidal Flux. If edge is se to True
        it load the edge value

        Examples
        -------
        >>> from tcv.diag import General
        >>> q95   = General.q(50882,plt=True)
        >>> qedge = General.q(50882,plt=True,edge=True)
        """
        if edge:
            Str = r'\results::q_edge'
            axlabel = r'q$_{edge}$'
        else:
            Str = r'\results::q_95'
            axlabel = r'q$_{95}$'

        data = tcv.shot(shot).tdi(Str)

        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :])
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(axlabel)
        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def kappa(shot, plt=False, edge=False):
        """
        Load the elongation (at 95% of the poloidal flux or edge)
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces
        edge : Boolean. If true it loads the valued at the edge
        Returns
        -------
        xray Data set with the elongation computed as default at
        95% of poloidal Flux. If edge is se to True
        it loads the edge value

        Examples
        -------
        >>> from tcv.diag import General
        >>> kappa95   = General.kappa(50882, plt=True)
        >>> kappaEdge = General.kappa(50882, plt=True, edge=True)
        """
        if edge:
            Str = r'\results::kappa_edge'
            axlabel = r'$\varepsilon_{edge}$'
        else:
            Str = r'\results::kappa_95'
            axlabel = r'$\varepsilon_{95}$'

        data = tcv.shot(shot).tdi(Str)

        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :])
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(axlabel)
        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def delta(shot, plt=False, edge=False, q95=True):
        """
        Load Triangularity (at 95% of the poloidal flux
        [default] or at the edge)
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces
        edge : Boolean (default is False). If True it
               loads value at the edge
        Returns
        -------
        xray Data set with the triangularity computed as default
        at 95% of poloidal Flux. If edge is se to True
        it load the edge value

        Examples
        -------
        >>> from tcv.diag import General
        >>> delta95   = General.delta(50882, plt=True)
        >>> deltaEdge = General.delta(50882, plt=True, edge=True)
        """
        if edge:
            q95 = False
            Str = r'\results::delta_edge'
            axlabel = r'$\delta_{edge}$'
        elif q95:
            Str = r'\results::delta_95'
            axlabel = r'$\delta_{95}$'

        data = tcv.shot(shot).tdi(Str)

        if plt:
            fig = mpl.pylab.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if data.values.shape[0] > 1:
                ax.plot(data[data.dims[0]].values, data.values)
            else:
                ax.plot(data[data.dims[0]].values, data.values[0, :])
            ax.set_title(r'Shot # ' + str(shot))
            ax.set_xlabel(r't [s]')
            ax.set_ylabel(axlabel)
        data = data.rename({'dim_0':'time'})
        return data

    @staticmethod
    def tedf(shot, plt=False, edge=False, q95=True):
        """
        Load Central temperature from Double Filter technique
        Parameters
        ----------
        shot : shot number
        plt  : Boolean (default is False). If True it
               plots the current time traces
        Returns
        -------
        xray Data set with the central temperature.
        Remember that there are 6 values (for different foil couples)
        It is generally convenient to use first one

        Examples
        -------
        >>> from tcv.diag import General
        >>> tedf   = General.tedf(50882, plt=True)
        """
        try:
            data = tcv.shot(shot).tdi(r'\results::te_x_a')

            if plt:
                fig = mpl.pylab.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                if data.values.shape[0] > 1:
                    ax.plot(data[data.dims[0]].values, data.values / 1e3)
                else:
                    ax.plot(data[data.dims[0]].values, data.values[0, :] / 1e3)
                ax.set_title(r'Shot # ' + str(shot))
                ax.set_xlabel(r't [s]')
                ax.set_ylabel(r'T_e [keV]')
            data = data.rename({'dim_0':'time'})
            return data
        except:
            print 'No data stored for Xte for this shot'
