from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# interpret F as DrawForward(1);
# interpret + as Yaw(90);
# interpret - as Yaw(-90);
# interpret n as Pitch(90);
# interpret & as Pitch(-90);
# interpret > as Roll(90);
# interpret < as Roll(-90);
# interpret | as Yaw(180);

A = r"B-F+CFC+F-D&FnD-F+&&CFC+F+B<<"
B = r"A&FnCFBnFnDnn-F-Dn|FnB|FCnFnA<<"
C = r"|Dn|FnB-F+CnFnA&&FA&FnC+F+BnFnD<<"
D = r"|CFB-F+B|FA&FnA&&FB-F+B|FC<<"

X = r"n<XFn<XFX-Fn>>XFX&F+>>XFX-F>X->"

# This Peano works infinitely - maybe not! but goes to at least n=3 iterations
P = r"P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP>>&F&P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP>>&F&P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP"

#these Peanos do not work
# Y = r"FF-F-FF+F+FFnFn+FF-F-FF+F+FF&F&+FF-F-FF+F+FF"  # Peano bottom to top
T = r"TFUFT+F+UFTFU-F-TFUFTnFn-ZFYFZ+F+YFZFY-F-ZFYFZ&F&-TFUFT+F+UFTFU-F-TFUFT"
U = T[::-1]
Y = r"YFZFY-F-ZFYFZ+F+YFZFYnFn+UFTFU-F-TFUFT+F+UFTFU&F&+YFZFY-F-ZFYFZ+F+YFZFY"
Z = Y[::-1]

# This guy needs dome testing
# R = r"RFRFR-F-RFRFR+F+RFRFRnFn+RFRFR-F-RFRFR+F+RFRFR&F&+RFRFR-F-RFRFR+F+RFRFR<<"
R = r"R>>FR>>FR>>+F+R>>FR>>FR>>+F+R>>FR>>FRnFn>>-R>>FR>>FR>>-F-R>>FR>>FR>>-F-R>>FR>>FRnFn>>+R>>FR>>FR>>+F+R>>FR>>FR>>+F+R>>FR>>FR"

d = {"A": A, "B": B, "C": C, "D": D, "X": X, "P": P, "Y": Y, "Z": Z, "T": T,
     "U": U, "R": R}


def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0,              0,             1]])


def roty(angle):
    return np.array([[np.cos(angle),  0, np.sin(angle)],
                     [0,              1,              0],
                     [-np.sin(angle), 0,  np.cos(angle)]])


def rotx(angle):
    return np.array([[1,             0,              0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])

# Each of these is a rotation in the frame of the local axis
# The local axis is rotated from the global axis (Identity)
# Ax_local = R*Ax_global  ==> R = Ax_local*Ax_global_inv = Ax_local
# RotH is a rotation about the local X axis
# RotH = R*RotX = Ax_local*RotX


def roth(local_axis, angle):
    return np.dot(local_axis, np.dot(rotx(angle), np.linalg.inv(local_axis)))


def rotl(local_axis, angle):
    return np.dot(local_axis, np.dot(roty(angle), np.linalg.inv(local_axis)))


def rotu(local_axis, angle):
    return np.dot(local_axis, np.dot(rotz(angle), np.linalg.inv(local_axis)))


def getEulerAngles(rotmatrix):
    """
    """
    sintheta = rotmatrix[2, 0]
    if abs(sintheta) != 1:
        theta = -np.arcsin(rotmatrix[2, 0])
        costheta = np.cos(theta)
        psi = np.arctan2(rotmatrix[2, 1]/costheta, rotmatrix[2, 2]/costheta)
        phi = np.arctan2(rotmatrix[1, 0]/costheta, rotmatrix[0, 0]/costheta)
    else:
        phi = 0
        if sintheta < 0:  # Positive case
            theta = np.pi/2.
            psi = phi + np.arctan2(rotmatrix[0, 1], rotmatrix[0, 2])
        else:
            theta = -np.pi/2.
            psi = -phi + np.arctan2(-rotmatrix[0, 1], -rotmatrix[0, 2])

    return psi, theta, phi


def iterate(inString):
    """
    """
    outString = []
    append = outString.append
    for char in inString:
        if char in d:
            append(d[char])
        else:
            append(char)

    return "".join(outString)


def generatePath(lstring, n=2, distance=10.):
    """
    """
    axis = np.eye(3)
    pos = [np.array([0, 0, 0])]

    def forward(axis, n=n, distance=distance):
        heading = np.dot(axis, np.array([1, 0, 0]))
        return [(ii+1)*distance*heading/n for ii in range(0, n)]

    charFunctions = {r"+": lambda axis: np.dot(rotu(axis, +np.pi/2.), axis),
                     r"-": lambda axis: np.dot(rotu(axis, -np.pi/2.), axis),
                     r"&": lambda axis: np.dot(rotl(axis, +np.pi/2.), axis),
                     r"n": lambda axis: np.dot(rotl(axis, -np.pi/2.), axis),
                     r">": lambda axis: np.dot(roth(axis, +np.pi/2.), axis),
                     r"<": lambda axis: np.dot(roth(axis, -np.pi/2.), axis),
                     r"|": lambda axis: np.dot(rotu(axis, np.pi), axis)}

    for char in lstring:
        if char == r"F":
            lastpos = pos[len(pos)-1]
            for point in forward(axis):
                pos.append(lastpos + point)
        elif char in charFunctions.keys():
            axis = charFunctions[char](axis)
        else:
            pass

    return pos


class Voxel(object):
    """
    Position, rotation and form of a DNA voxel.

    Members:
    psi, theta, phi: Euler rotations about the X, Y and Z axes
                     (XYZ rotation order)

    type: code corresponding to geometrical shape
    pos: XYZ position

    Methods:
    toText: Print a textual representation of the voxel as
            KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI
    """
    types = {'straight': 1, 'straighttwist': 2, 'turn': 3, 'turntwist': 4}
    types_inverse = {v: k for (k, v) in types.items()}
    defaultPrincipal = np.array([1, 0, 0])
    defaultHeading = np.array([0, 0, 1])
    defaultOrtho = np.cross(defaultPrincipal, defaultHeading)
    defaultAxis = np.array([defaultPrincipal, -defaultOrtho, defaultHeading])
    defaultAxis = np.transpose(defaultAxis)
    defaultAxisInv = np.linalg.inv(defaultAxis)

    def __init__(self, pos, inHeading, inPrincipal, outHeading, outPrincipal):
        """
        Voxel(pos, inHeading, inPrincipal, outHeading, outPrincipal)

        Identifies the form of the DNA voxel that corresponds to specified
        input and output vectors.
        """
        # Clean and vet input
        pos = np.around(pos, decimals=8)
        inHeading = np.around(inHeading, decimals=8)
        inPrincipal = np.around(inPrincipal, decimals=8)
        outHeading = np.around(outHeading, decimals=8)
        outPrincipal = np.around(outPrincipal, decimals=8)

        assert (inHeading != inPrincipal).any(), \
            "degenerate entry vectors: " + str(inHeading) + str(inPrincipal)
        assert (outHeading != outPrincipal).any(), \
            "degenerate exit vectors: " + str(inPrincipal) + str(outPrincipal)

        self.inHeading = inHeading/np.linalg.norm(inHeading)
        self.inPrincipal = inPrincipal/np.linalg.norm(inPrincipal)
        self.outHeading = outHeading/np.linalg.norm(outHeading)
        self.outPrincipal = outPrincipal/np.linalg.norm(outPrincipal)
        self.pos = pos

        sameHeading = (self.inHeading == self.outHeading).all()
        samePrincipal = ((self.inPrincipal == self.outPrincipal).all() or
                         (self.inPrincipal == -self.outPrincipal).all())

        if sameHeading:
            if samePrincipal:
                self.type = self.types['straight']
            else:
                self.type = self.types['straighttwist']
        else:
            if samePrincipal:
                self.type = self.types['turn']
            else:
                self.type = self.types['turntwist']

        # Now we need to define the euler rotations of the pixel box
        # We assume that "No Rotation" involves the path entering with a +z
        # direction heading with the principal axis being the +x direction
        if self.type in [self.types['straight'], self.types['straighttwist']]:
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

    def toText(self):
        """
        """
        l = [self.types_inverse[self.type]] + map(str, list(self.pos)) + \
            map(str, [self.psi, self.theta, self.phi])
        return " ".join(l) + "\n"


class VoxelisedFractal(object):
    """
    Class containing a voxelised representation of a fractal
    """
    def __init__(self):
        """
        """
        self.fractal = []

    def __len__(self):
        return self.fractal.__len__()

    def toText(self):
        """
        """
        output = "#KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI\n"
        text = [output] + [voxel.toText() for voxel in self.fractal]
        return "".join(text)

    def toPlot(self, refine=0, batch=False):
        """
        fig = toPlot(refine=0, batch=False)
        Create a matplotlib figure instance of this fractal

        kwargs
        ---
        refine: points to plot in between voxels (more points = clearer path)
        batch: True to suppress automatic display of the figure
        """
        pts = [vox.pos for vox in self.fractal]

        refinedpts = []
        for ii in range(0, len(pts)-1):
            refinedpts.append(pts[ii])
            step = (pts[ii+1] - pts[ii])/(refine+1)
            for jj in range(1, refine+1):
                refinedpts.append(pts[ii] + step*jj)
        refinedpts.append(pts[len(pts)-1])

        pts = np.array(refinedpts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

        if batch is not True:
            fig.show()

        return fig

    @staticmethod
    def makeVoxel(prevVoxel, currpos, nextpos):
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
            return Voxel(currpos, prevVoxel.outHeading, prevVoxel.outPrincipal,
                         secondChange, perp)
        else:
            return Voxel(currpos, prevVoxel.outHeading, prevVoxel.outPrincipal,
                         prevVoxel.outHeading, prevVoxel.outPrincipal)

    @classmethod
    def fromSeed(cls, seed, iterations, distance=1):
        """
        """
        for n in range(iterations):
            seed = iterate(seed)
        return cls.fromLString(seed, distance=distance)

    @classmethod
    def fromLString(cls, lstring, distance=1):
        """
        """
        path = generatePath(lstring, n=1, distance=distance)
        lastpos = 2*path[len(path)-1] - path[len(path)-2]
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
        zeroposition = first - distance*zeroheading
        zeroprincipal = np.array([1, 0, 0])
        if (np.around(zeroheading, 8) == zeroprincipal).all():
            zeroprincipal = np.array([0, 1, 0])
        zeroVoxel = Voxel(zeroposition, zeroheading, zeroprincipal,
                          zeroheading, zeroprincipal)

        vf.fractal = [cls.makeVoxel(zeroVoxel, first, second)]

        for ii in range(1, len(path) - 1):
            vf.fractal.append(cls.makeVoxel(vf.fractal[ii-1], arrpath[ii],
                                            arrpath[ii+1]))
            # print np.around(arrpath[ii], 3)
        # print "Path Length: ", len(vf.fractal)

        return vf
