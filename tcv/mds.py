""" TCV MDS datasource """

from collections import OrderedDict
from itertools import cycle

import MDSplus as mds
import xarray as xray

from .datasource import DataSource


class MDSConnection(DataSource):
    """ Generic MDSPlus datasource """

    _MAX_DIMS = 5

    def __init__(self, shot, tree, server):
        """
        Parameters
        ----------
        shot : int
            Shot number.
        tree : str
            MDSplus tree name.
        server : str
            Address of the MDSplus server
        """
        self.shot = shot
        self.tree = tree
        self.server = server

#        self._conn = mds.Connection(server)
#        self._conn.openTree(tree, shot)
        self._conn = mds.Tree(self.tree, self.shot)

    def tdi(self, cmd, *args, **kwargs):
        """
        Execute a TDI command.

        Parameters
        ----------
        cmd : str
            The TDI command to execute.
        args : variable length argument list
            Arguments to be passed to the TDI command

        Keyword arguments
        -----------------
        dims : str or sequence of str, optional
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omited, dimension names
            are taken from the MDS tree (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        """
        return self._as_xray(cmd, *args, **kwargs)

    def close(self):
        self._conn.quit()

    def _as_xray(self, query, *args, **kwargs):
        """ Read one signal through the connection """

        # arrays are returned in column-major (Fortran) order so we use .T
        data = mds.Data.compile(query, *args).evaluate().data().T
        dims = kwargs.get('dims', None)

        if dims:
            if not isinstance(dims, (list, tuple)):
                dims = [dims]
            assert len(dims) == data.ndim, \
                "Must provide dimension name for each data dimension."
        else:
            dims = cycle([''])

        coords = OrderedDict()
        for i, dim_name in zip(xrange(MDSConnection._MAX_DIMS), dims):
            try:
                coords.update(self._get_dim(i, query, dim_name, *args))
            except mds.MdsException:
                break

        attrs = {}
        attrs.update(self._get_units(query))
        attrs.update({'shot': self.shot, 'query': query})

        return xray.DataArray(data, coords=coords.values(),
                              dims=coords.keys(), name=query,
                              attrs=attrs)

    def _get_dim(self, i, query, name, *args):
        """ Get i-th dimension of the specified query. """

        dim_of = r'dim_of({}, {})'.format(query, i)
        name_of = r'name_of({})'.format(dim_of)

        try:
            dim_name = mds.Data.compile(name_of, *args).evaluate().data()
        except mds.MdsException:
            dim_name = name if name else 'dim_{}'.format(i)

        return {dim_name: mds.Data.compile(dim_of, *args).evaluate().data()}

    def _get_units(self, query):
        """ Get the physical units of the specified query """

        try:
            units = mds.Data.compile(r'units_of({})'.format(query)).evaluate().data()
            return {'units': units}
        except mds.MdsException:
            return {}
