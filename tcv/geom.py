# pylint: disable=invalid-name, no-member
"""
TCV geometry and vessel properties
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator
import MDSplus
import tcv


class StaticParameters(object):
    """ TCV static parameters """

    def __init__(self, coordinates):
        self._coordinates = coordinates

    def get_coordinates(self, part, coord_names=None):
        """
        Get the coordinates of a certain vessel part.
        """
        if coord_names is None:
            return self._coordinates[part]

        names = coord_names.split(',')
        names = [str.strip(x) for x in names]

        ret = [self._coordinates[part][key] for key in names]

        if len(ret) == 1:
            ret = ret[0]
        else:
            ret = tuple(ret)
        return ret

    @classmethod
    def from_tree(cls, shotnum):
        """ Read the vessel's parameters from the static MDS nodes """
        Tree = MDSplus.Tree('tcv_shot',shotnum)
        vessel = {}
#        with tcv.shot(shotnum) as conn:
        for key, query in cls._vessel_nodes().iteritems():
            vessel[key] = MDSplus.Data.execute(query).getvalue().data()

        tiles = {}
        tiles['R'] = np.array([
            1.1360, 1.1360, 1.1120, 1.0880, 1.0640, 1.0399, 1.0159, 0.9919,
            0.9679, 0.6707, 0.6240, 0.6240, 0.6240, 0.6724, 0.9679, 1.1360,
            1.1360
        ])
        tiles['Z'] = np.array([
            0, 0.5494, 0.5781, 0.6067, 0.6354, 0.6640, 0.6927, 0.7213, 0.7500,
            0.7500, 0.7033, 0, -0.7033, -0.7500, -0.7500, -0.5494, 0.0000
        ])

        return cls({'vessel': vessel, 'tiles': tiles})

    @staticmethod
    def _vessel_nodes():
        """ Return the static MDS nodes containing the vessel parameters """

        nodes = {
            'R_in': 'static("r_v:in")',
            'R_out': 'static("r_v:out")',
            'Z_in': 'static("z_v:in")',
            'Z_out': 'static("z_v:out")'}
        return nodes


class VesselDrawer(object):
    def __init__(self, coordinates):
        """
        Input
        -----
        coordinates : dict
            Dictionary with the following structure
                'vessel': dict with keys 'R_in', 'Z_in', 'R_out', 'Z_out'
                'tiles': dict with keys 'R', 'Z'
        """
        self.coordinates = coordinates
        self._tiles_shown = False

    @classmethod
    def from_static(cls, shotnum=0):
        static = StaticParameters.from_tree(shotnum)
        coords = {'vessel': static.get_coordinates('vessel'),
                  'tiles': static.get_coordinates('tiles')}
        return cls(coords)

    def render(self, ax=None, tiles=True):
        ax = ax or plt.gca()
        ax.add_patch(self._get_vessel_patch())
        if tiles and self.has_tiles_coordinates():
            ax.add_patch(self._get_tiles_patch())
            self._tiles_shown = True

    def get_coordinates(self, key):
        return self.coordinates.get(key)

    def has_tiles_coordinates(self):
        return 'tiles' in self.coordinates

    def get_patches(self):
        patches = [self._get_vessel_patch()]
        if self.get_coordinates('tiles') is not None:
            patches.append(self._get_tiles_patch())
        return patches

    def _inside(self):
        coords = self.get_coordinates('vessel')
        vertices = [r for r in zip(coords['R_in'], coords['Z_in'])]
        vertices.append(vertices[0])
        codes = [Path.MOVETO] + (len(vertices)-1) * [Path.LINETO]
        return vertices, codes

    def _outside(self):
        coords = self.get_coordinates('vessel')
        vertices = [r for r in zip(coords['R_out'], coords['Z_out'])][::-1]
        vertices.append(vertices[0])
        codes = [Path.MOVETO] + (len(vertices)-1) * [Path.LINETO]
        return vertices, codes

    def _tiles(self):
        coords = self.get_coordinates('tiles')
        vertices = [r for r in zip(coords['R'], coords['Z'])][::-1]
        vertices.append(vertices[0])
        codes = [Path.MOVETO] + (len(vertices)-1) * [Path.LINETO]
        return vertices, codes

    def _get_vessel_patch(self):
        vertices_in, codes_in = self._inside()
        vertices_out, codes_out = self._outside()
        vessel_path = Path(vertices_in + vertices_out, codes_in + codes_out)
        vessel_patch = PathPatch(vessel_path, facecolor=(0.6, 0.6, 0.6),
                                 edgecolor='black')
        return vessel_patch

    def _get_tiles_patch(self):
        vertices_in, codes_in = self._tiles()
        vertices_out, codes_out = self._inside()
        tiles_path = Path(vertices_in + vertices_out, codes_in + codes_out)
        tiles_patch = PathPatch(tiles_path, facecolor=(0.75, 0.75, 0.75),
                                edgecolor='black')

        return tiles_patch

    def _transparent_inside_patch(self):
        if self._tiles_shown:
            vertices, codes = self._tiles()
        else:
            vertices, codes = self._inside()
        path = Path(vertices, codes)
        return PathPatch(path, alpha=0)

    def clip_collections_inside(self, ax):
        vessel_in = self._transparent_inside_patch()
        ax.add_patch(vessel_in)
        for c in ax.collections:
            c.set_clip_path(vessel_in)


def tcvview(shotnum, time, vessel=True, ports=False,
            col='red', tiles=True):
    """
    Popular way to display TCV.

    Parameters
    ----------
    shotnum : int, or MDSConnection
        Shot number or an open MDS connection instance.
    time : float
        Time of the equilibrium.
    vessel : bool, optional
        Draw vessel.
    ports : bool, optional
        Draw ports
    tiles : bool, optional
        Draw tiles.

    Example
    -------
    >>> import tcv
    >>> tcv.tcvview(42660, 1)
    """
    ax = plt.gca()

    from . equilibrium import LiuqeEquilibrium

    eq = LiuqeEquilibrium.fromshot(shotnum, time)

    if ports:
        PortsDrawer().render()

    if vessel:
        vd = VesselDrawer.from_static(shotnum)
        vd.render(ax, tiles)

    r, z, psi = eq.get_psi_contours()
    r0, z0 = eq.magnetic_axis
    levels_core = np.linspace(0, 10, 101)
    levels_sol = np.linspace(-10, 0, 101)

    ax.contour(r, z, psi.T, levels=levels_core, colors=col)
    ax.contour(r, z, psi.T, levels=levels_sol, colors=col, linestyles=':')
    ax.plot([r0], [z0], 'r+')
    ax.set_aspect('equal')

    if vessel:
        vd.clip_collections_inside(ax)
    _add_shot_time(eq.shot, eq.time, ax)

    ax.xaxis.set_major_locator(MaxNLocator(4))

    if plt.isinteractive():
        ax.figure.canvas.draw()


def _add_shot_time(shot, time, ax):
    text = r'$\#%d\ %1.3f\ \mathrm{s}$' % (shot, time)
    ax.text(1.0, 0, text, rotation=270, transform=ax.transAxes,
            va='bottom', ha='left', size='small')


def _add_plasma_parameters(connection, time, ax):
    text = r'$I_\mathrm{p}=999\ \mathrm{kA}$'
    ax.text(0.5, 1.0, text, transform=ax.transAxes,
            va='bottom', ha='center', size='small')


class PortsDrawer(object):
    def get_ports_coordinates(self):
        Hportx = np.array([
            0.04, 0, 0, 0.022, 0.046, 0.245, 0.245, 0.27, 0.27, 0.04
        ])
        Hporty = np.array([
            0.06, 0.06, 0.115, 0.115, 0.08, 0.08, 0.126, 0.126, 0.064, 0.064
        ])

        pm1y = 0.455
        pt1y = pm1y + Hporty
        pb1y = pm1y - Hporty
        p1x = 1.16 + Hportx

        pm2y = -0.0025
        pt2y = pm2y + Hporty
        pb2y = pm2y - Hporty
        p2x = 1.16 + Hportx

        pm3y = -0.46
        pt3y = pm3y + Hporty
        pb3y = pm3y - Hporty
        p3x = 1.16 + Hportx

        Vportx = np.array([
            0.035, 0.035, 0.068, 0.068, 0.054, 0.054, 0.075, 0.075, 0.04, 0.04
        ])
        Vporty = np.array([
            0.045, 0, 0, 0.045, 0.059, 0.069, 0.069, 0.09, 0.09, 0.045
        ])
        Vporty1 = np.array([
            0.045, -0.059157, -0.086867, 0.045, 0.059, 0.069, 0.069, 0.09,
            0.09, 0.045
        ])

        pm4x = 0.715
        pl4x = pm4x - Vportx
        pr4x = pm4x + Vportx
        p4y = 0.77 + Vporty

        pm5x = 0.715
        pl5x = pm5x - Vportx
        pr5x = pm5x + Vportx
        p5y = -0.77 - Vporty

        pm6x = 0.88
        pl6x = pm6x - Vportx
        pr6x = pm6x + Vportx
        pl6y = -0.77 - Vporty
        # pm6y = -0.77 - Vporty
        pr6y = -0.77 - Vporty

        # pm7x = 0.88
        pl7x = pm6x - Vportx
        pr7x = pm6x + Vportx
        pl7y = 0.77 + Vporty
        # pm7y = 0.77 + Vporty
        pr7y = 0.77 + Vporty

        pm8x = 1.045
        pl8x = pm8x - Vportx
        pr8x = pm8x + Vportx
        pl8y = -0.77 - Vporty
        # pm8y = -0.77 - Vporty
        pr8y = -0.77 - Vporty1

        pm9x = 1.045
        pl9x = pm9x - Vportx
        pr9x = pm9x + Vportx
        pl9y = 0.77 + Vporty
        # pm9y = 0.77 + Vporty
        pr9y = 0.77 + Vporty1

        p1 = self._vertices_codes(p1x, pt1y, p1x, pb1y)
        p2 = self._vertices_codes(p2x, pt2y, p2x, pb2y)
        p3 = self._vertices_codes(p3x, pt3y, p3x, pb3y)

        p4 = self._vertices_codes(pl4x, p4y, pr4x, p4y)
        p5 = self._vertices_codes(pl5x, p5y, pr5x, p5y)
        p6 = self._vertices_codes(pl6x, pl6y, pr6x, pr6y)
        p7 = self._vertices_codes(pl7x, pl7y, pr7x, pr7y)
        p8 = self._vertices_codes(pl8x, pl8y, pr8x, pr8y)
        p9 = self._vertices_codes(pl9x, pl9y, pr9x, pr9y)

        return [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def _vertices_codes(self, x1, y1, x2, y2):
        vertices1 = [r for r in zip(x1, y1)]
        vertices1.append(vertices1[0])
        codes1 = [Path.MOVETO] + (len(vertices1)-1) * [Path.LINETO]
        vertices2 = [r for r in zip(x2, y2)]
        vertices2.append(vertices2[0])
        codes2 = [Path.MOVETO] + (len(vertices2)-1) * [Path.LINETO]

        vertices = vertices1 + vertices2
        codes = codes1 + codes2

        return vertices, codes

    def get_patch(self):
        codes, vertices = [], []
        for p in self.get_ports_coordinates():
            v, c = p
            codes += c
            vertices += v

        path = Path(vertices, codes)
        return PathPatch(path, facecolor=(0.6, 0.6, 0.6), edgecolor='black')

    def render(self, ax=None):
        ax = ax or plt.gca()
        ax.add_patch(self.get_patch())
