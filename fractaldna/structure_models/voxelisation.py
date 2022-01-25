from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from . import hilbert

try:
    from mayavi import mlab

    maya_imported = True
except ImportError:
    maya_imported = False
    print("Could not import mayavi libraries, 3d plotting is disabled")
    print("MayaVi may need Python2")


def rot_ax_angle(axis: Union[List, np.array], angle: float) -> np.array:
    """Build the rotation matrix for a given rotation around an axis

    :param axis: Rotation axis (3-vector)
    :param angle: rotation angle (radians)

    :returns: The rotation matrix
    """
    ax = axis / np.sqrt(np.sum(np.array(axis) ** 2))
    ux = ax[0]
    uy = ax[1]
    uz = ax[2]

    c = np.cos(angle)
    s = np.sin(angle)

    xx = c + ux ** 2 * (1 - c)
    xy = ux * uy * (1 - c) - uz * s
    xz = ux * uz * (1 - c) + uy * s

    yx = uy * ux * (1 - c) + uz * s
    yy = c + uy ** 2 * (1 - c)
    yz = uy * uz * (1 - c) - ux * s

    zx = uz * ux * (1 - c) - uy * s
    zy = uz * uy * (1 - c) + ux * s
    zz = c + uz ** 2 * (1 - c)

    return np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])


def getEulerAngles(rotmatrix: np.array) -> Tuple[float, float, float]:
    """Get the euler angles from a rotation matrix

    :param rotmatrix: 3x3 rotation matrix
    :return: Euler psi angle
    """
    sintheta = rotmatrix[2, 0]
    if abs(sintheta) != 1:
        theta = -np.arcsin(rotmatrix[2, 0])
        costheta = np.cos(theta)
        psi = np.arctan2(rotmatrix[2, 1] / costheta, rotmatrix[2, 2] / costheta)
        phi = np.arctan2(rotmatrix[1, 0] / costheta, rotmatrix[0, 0] / costheta)
    else:
        phi = 0
        if sintheta < 0:  # Positive case
            theta = np.pi / 2.0
            psi = phi + np.arctan2(rotmatrix[0, 1], rotmatrix[0, 2])
        else:
            theta = -np.pi / 2.0
            psi = -phi + np.arctan2(-rotmatrix[0, 1], -rotmatrix[0, 2])

    return psi, theta, phi


class Voxel:
    """
    Position, rotation and form of a DNA placement voxel.

    The principal axes are used to determine if the DNA would
    undergo a 90˚ rotation as it passes through the voxel.

    One way to think of them is to consider that the path of your
    DNA or curve through the voxel is along a continuous z-axis,
    perpendicular to an X-Y plane. The Principal Axes could be the
    vectors that describe the direction of the X-axis.

    Class Notes:
    psi, theta, phi: Euler rotations about the X, Y and Z axes
    (XYZ rotation order)

    type: code corresponding to geometrical shape
    pos: XYZ position

    Voxel(pos, inHeading, inPrincipal, outHeading, outPrincipal)

    :param pos: XYZ-position of voxel
    :param inHeading: vector of the DNA heading "into" the voxel.
    :param inPrincipal: vector of the DNA's principal axis at the
        entrace to the voxel
    :param outHeading: vector of the DNA heading "out" of the voxel.
    :param outPrincipal: vector of the DNA's principal axis at the
        exit to the voxel
    """

    types = {"straight": 1, "straighttwist": 2, "turn": 3, "turntwist": 4}
    types_inverse = {v: k for (k, v) in types.items()}
    defaultPrincipal = np.array([1, 0, 0])
    defaultHeading = np.array([0, 0, 1])
    defaultOrtho = np.cross(defaultPrincipal, defaultHeading)
    defaultAxis = np.array([defaultPrincipal, -defaultOrtho, defaultHeading])
    defaultAxis = np.transpose(defaultAxis)
    defaultAxisInv = np.linalg.inv(defaultAxis)

    def __init__(
        self,
        pos: np.array,
        inHeading: np.array,
        inPrincipal: np.array,
        outHeading: np.array,
        outPrincipal: np.array,
    ):
        """
        Constructor
        """
        # Clean and vet input
        pos = np.around(pos, decimals=8)
        inHeading = np.around(inHeading, decimals=8)
        inPrincipal = np.around(inPrincipal, decimals=8)
        outHeading = np.around(outHeading, decimals=8)
        outPrincipal = np.around(outPrincipal, decimals=8)

        assert (inHeading != inPrincipal).any(), (
            "degenerate entry vectors: " + str(inHeading) + str(inPrincipal)
        )
        assert (outHeading != outPrincipal).any(), (
            "degenerate exit vectors: " + str(inPrincipal) + str(outPrincipal)
        )

        self.inHeading = inHeading / np.linalg.norm(inHeading)
        self.inPrincipal = inPrincipal / np.linalg.norm(inPrincipal)
        self.outHeading = outHeading / np.linalg.norm(outHeading)
        self.outPrincipal = outPrincipal / np.linalg.norm(outPrincipal)
        self.pos = pos

        sameHeading = (self.inHeading == self.outHeading).all()
        samePrincipal = (self.inPrincipal == self.outPrincipal).all() or (
            self.inPrincipal == -self.outPrincipal
        ).all()

        if sameHeading:
            if samePrincipal:
                self.type = self.types["straight"]
            else:
                self.type = self.types["straighttwist"]
        else:
            if samePrincipal:
                self.type = self.types["turn"]
            else:
                self.type = self.types["turntwist"]

        # Now we need to define the euler rotations of the pixel box
        # We assume that "No Rotation" involves the path entering with a +z
        # direction heading with the principal axis being the +x direction
        if self.type in [self.types["straight"], self.types["straighttwist"]]:
            paxis = self.inPrincipal
        else:
            paxis = self.outHeading

        self.orth = np.cross(paxis, self.inHeading)
        self.axis = np.array([paxis, -self.orth, self.inHeading])
        self.axis = np.transpose(self.axis)
        self.rotation = np.dot(self.axis, self.defaultAxisInv)

        # Euler angles
        # Using (x, y, z) rotations = (psi, theta, phi)
        self.psi, self.theta, self.phi = getEulerAngles(self.rotation)

    def to_text(self, sep: str = " ") -> str:
        """Print a textual representation of the voxel as
        KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI
        """
        if len(sep) == 0:
            raise ValueError("Separator cannot be a zero-length string")
        l = (
            [self.types_inverse[self.type]]
            + list(map(str, list(self.pos)))
            + list(map(str, [self.psi, self.theta, self.phi]))
        )
        return sep.join(l)

    def to_series(self):
        """Return the voxel as a pandas series"""
        return pd.Series(
            {
                "KIND": self.types_inverse[self.type],
                "POS_X": self.pos[0],
                "POS_Y": self.pos[1],
                "POS_Z": self.pos[2],
                "EUL_PSI": self.psi,
                "EUL_THETA": self.theta,
                "EUL_PHI": self.phi,
            }
        )


class VoxelisedFractal:
    """
    Class containing a voxelised representation of a fractal

    Typically created using the VoxelisedFractal.fromSeed class method

    fractal = VoxelisedFractal.fromSeed('X', 1)
    fractal.fractal contains a list of 'Voxels' representing the DNA Path

    """

    def __init__(self):
        """Constructor"""
        self.fractal = []

    def __len__(self) -> int:
        return self.fractal.__len__()

    def to_text(self, sep: str = " ", comment: str = "#"):
        """Return a textual representation of the fractal as
            KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI

        :param sep: string to use to separate fields (default space)
        :param comment: string to use to denote comment in first line
            (default hash: #)
        """
        if len(sep) == 0:
            raise ValueError("Separator cannot be a zero-length string")

        output = "IDX KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI"
        output = output.replace(" ", sep)
        text = [comment + output] + [
            sep.join([str(idx), voxel.to_text(sep=sep)])
            for idx, voxel in enumerate(self.fractal)
        ]
        return "\n".join(text)

    def to_frame(self):
        """Convert voxelised representation to data frame"""
        rows = []
        for idx, voxel in enumerate(self.fractal):
            ss = voxel.to_series()
            ss["IDX"] = idx
            rows.append(ss)
        df = pd.DataFrame(rows)
        return df[
            [
                "IDX",
                "KIND",
                "POS_X",
                "POS_Y",
                "POS_Z",
                "EUL_PSI",
                "EUL_THETA",
                "EUL_PHI",
            ]
        ]

    def to_plot(self, refine: int = 0, batch: bool = False) -> plt.figure:
        """
        Create a matplotlib figure instance of this fractal

        fig = toPlot(refine=0, batch=False)
        :param refine: points to plot in between voxels (more points = clearer path)
        :param batch: True to suppress automatic display of the figure
        """
        pts = [vox.pos for vox in self.fractal]

        refinedpts = []
        for ii in range(0, len(pts) - 1):
            refinedpts.append(pts[ii])
            step = (pts[ii + 1] - pts[ii]) / (refine + 1)
            for jj in range(1, refine + 1):
                refinedpts.append(pts[ii] + step * jj)
        refinedpts.append(pts[len(pts) - 1])

        pts = np.array(refinedpts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])

        if batch is not True:
            fig.show()

        return fig

    def to_pretty_plot(
        self, refine: int = 10, mayavi: bool = False, mask: Callable = None
    ):
        """Create a matplotlib figure instance of this fractal, with curved lines
        rather than hard corners

        fig = to_pretty_plot(refine=10, batch=False)

        :param refine: points to plot in between voxels (more points = clearer path)
        :param mayavi: make plot with MayaVI
        :param mask: Callable function that returns true for voxels that should be
            plotted. mask is a function that should take a position three vector and
            return boolean True/False for any position.
        """
        if mask is not None:
            if not callable(mask):
                raise ValueError("Mask should be callable")
        pts = [vox.pos for vox in self.fractal]

        # replace pts with an array containing the sides of boxes rather than
        # the centres of boxes. We are then going to interp curves b/w points
        midpoints = [0.5 * (ii + jj) for ii, jj in zip(pts[0:-1], pts[1:])]

        refinedpts = [pts[0]]
        for ii in range(0, len(midpoints) - 1):
            # print(ii)
            step = 1 / (refine)
            entry_point = midpoints[ii]
            exit_point = midpoints[ii + 1]
            entry_normal = midpoints[ii] - pts[ii]
            exit_normal = midpoints[ii + 1] - pts[ii + 1]
            interp = self._interpolator(
                entry_point, entry_normal, exit_point, exit_normal
            )
            for jj in range(0, refine):
                # print(jj*step)
                refinedpts.append(interp(step * jj))
        refinedpts.append(pts[-1])

        pts = np.array(refinedpts)
        idx = np.round(np.arange(len(pts)) / len(pts), 3)
        pts = np.concatenate([pts, idx.reshape([len(idx), 1])], axis=1)
        if mayavi:
            if mask is not None:
                # iterate over plot_points to find acceptable points
                plot_points = [ii for (ii, pos) in enumerate(pts) if mask(pos)]
                grouped_plot_points = []
                while len(plot_points) > 0:
                    end_point = plot_points.pop()
                    grouped_plot_points
                    current_point = end_point
                    while plot_points[-1] == current_point - 1:
                        current_point = plot_points.pop()
                        if len(plot_points) == 0:
                            break
                    grouped_plot_points.append((current_point, end_point))
                pts = [pts[start : end + 1] for start, end in grouped_plot_points]

            else:
                pts = [pts]
            fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
            for arr in pts:
                assert np.all(arr[:, 3] >= 0) and np.all(
                    arr[:, 3] <= 1
                ), """color value was {}, outside acceptable range
                    [0, 1]""".format(
                    arr[:, 3]
                )
                mlab.plot3d(
                    arr[:, 0],
                    arr[:, 1],
                    arr[:, 2],
                    arr[:, 3],
                    # color=(0., .8, 0),
                    colormap="Spectral",
                    tube_radius=0.1,
                    vmin=0.0,
                    vmax=1.0,
                )
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])

        return fig

    def center_fractal(self) -> None:
        """Center the fractal around (x, y, z) = (0, 0, 0)"""
        minvals = np.array([np.inf, np.inf, np.inf])
        maxvals = np.array([-np.inf, -np.inf, -np.inf])

        # identify max/min values
        for voxel in self.fractal:
            for (ii, (minv, v, maxv)) in enumerate(zip(minvals, voxel.pos, maxvals)):
                if v < minv:
                    minvals[ii] = v
                elif v > maxv:
                    maxvals[ii] = v

        # transform
        transform = -minvals - (maxvals - minvals) / 2.0
        for voxel in self.fractal:
            oldpos = voxel.pos
            voxel.pos = oldpos + transform

        return None

    def _interpolator(self, point_entry, norm_entry, point_exit, norm_exit):
        """Interpolator for pretty_plot"""
        # 1. case 1, norm_entry = norm_exit
        if (norm_entry == norm_exit).all():
            interp = lambda x: point_entry + x * (point_exit - point_entry)
            return interp
        # 2. Case 2, circular interpolation
        # 2a. find centre of circle
        # pdb.set_trace()
        norm_plane = np.cross(norm_entry, norm_exit)
        d1 = -np.dot(point_entry, norm_entry)
        d2 = -np.dot(point_exit, norm_exit)
        d3 = -np.dot(point_entry, norm_plane)
        centre = -(
            d1 * np.cross(norm_exit, norm_plane)
            + d2 * np.cross(norm_plane, norm_entry)
            + d3 * np.cross(norm_entry, norm_exit)
        ) / (np.dot(norm_entry, np.cross(norm_exit, norm_plane)))
        v_init = point_entry - centre
        v_final = point_exit - centre
        rotation_axis = np.cross(v_init, v_final)
        rotation_axis /= np.linalg.norm(rotation_axis)  # unit vector

        # function to change the magnitude of the vector
        mag = lambda x: np.linalg.norm(v_init) + x * (
            np.linalg.norm(v_final) - np.linalg.norm(v_init)
        )

        vec = lambda x: np.dot(rot_ax_angle(rotation_axis, x * np.pi / 2.0), v_init)

        return lambda x: centre + 2 * mag(x) * vec(x)

    @staticmethod
    def _makeVoxel(prevVoxel: Voxel, currpos: np.array, nextpos: np.array) -> Voxel:
        """Make the next voxel in a chain.

        :param prevVoxel: Previous Voxel in Chain
        :param currpos: Position (centre) of the voxel to make
        :param nexpos: Position (centre) of the next voxel to make
        """
        # clean and vet output
        currpos = np.around(currpos, 8)
        nextpos = np.around(nextpos, 8)
        prevpos = prevVoxel.pos
        firstChange = currpos - prevpos
        secondChange = nextpos - currpos
        perp = np.cross(firstChange, secondChange)
        turn = perp.any()
        if turn:
            perp = perp.round()
            return Voxel(
                currpos,
                prevVoxel.outHeading,
                prevVoxel.outPrincipal,
                secondChange,
                perp,
            )
        else:
            return Voxel(
                currpos,
                prevVoxel.outHeading,
                prevVoxel.outPrincipal,
                prevVoxel.outHeading,
                prevVoxel.outPrincipal,
            )

    @classmethod
    def fromSeed(cls, seed: str, iterations: int, distance: float = 1):
        """Make a voxelised fractal from an L-String seed.

        The available L-Strings are described in fractaldna.structure_models.hilbert

        :param seed: Seed L-String
        :param iterations: number of times to iterate seed
        :param distance: distance between voxels

        :returns: Voxelised Fractal representation
        """
        for n in range(iterations):
            seed = hilbert.iterate_lstring(seed)
        return cls.fromLString(seed, distance=distance)

    @classmethod
    def fromLString(cls, lstring, distance=1):
        """Convert an L-String into a VoxelisedFractal

        :param lstring: L-String to convert
        :param distance: distance between voxels

        :returns: Voxelised Fractal representation
        """
        path = hilbert.generate_path(lstring, n=1, distance=distance)
        lastpos = 2 * path[len(path) - 1] - path[len(path) - 2]
        path.append(lastpos)

        vf = cls()

        arrpath = np.array(path)

        mins = np.zeros(3)  # array [xmin, ymin, zmin]
        maxs = np.zeros(3)  # array [xmax, ymax, zmax]
        lens = np.zeros(3)

        for ii in range(3):
            mins[ii] = min(arrpath[:, ii])
            maxs[ii] = max(arrpath[:, ii])
            lens[ii] = maxs[ii] - mins[ii]

        first = arrpath[0]
        second = arrpath[1]
        zeroheading = second - first
        zeroposition = first - distance * zeroheading
        zeroprincipal = np.array([1, 0, 0])
        if (np.around(zeroheading, 8) == zeroprincipal).all():
            zeroprincipal = np.array([0, 1, 0])
        zeroVoxel = Voxel(
            zeroposition, zeroheading, zeroprincipal, zeroheading, zeroprincipal
        )

        vf.fractal = [cls._makeVoxel(zeroVoxel, first, second)]

        for ii in range(1, len(path) - 1):
            vf.fractal.append(
                cls._makeVoxel(vf.fractal[ii - 1], arrpath[ii], arrpath[ii + 1])
            )
            # print np.around(arrpath[ii], 3)
        # print "Path Length: ", len(vf.fractal)

        return vf
