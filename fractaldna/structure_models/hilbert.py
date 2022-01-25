from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # NOQA

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


# Legacy Peano Curves
# This Peano works infinitely - maybe not! but goes to at least n=3 iterations
P = r"P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP>>&F&P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP>>&F&P>>FP>>FP>>+F+P>>FP>>FP>>+F+P>>FP>>FP"  # NOQA

# # These Peanos do not work
# # Y = r"FF-F-FF+F+FFnFn+FF-F-FF+F+FF&F&+FF-F-FF+F+FF"  # Peano bottom to top
# T = r"TFUFT+F+UFTFU-F-TFUFTnFn-ZFYFZ+F+YFZFY-F-ZFYFZ&F&-TFUFT+F+UFTFU-F-TFUFT"
# U = T[::-1]
# Y = r"YFZFY-F-ZFYFZ+F+YFZFYnFn+UFTFU-F-TFUFT+F+UFTFU&F&+YFZFY-F-ZFYFZ+F+YFZFY"
# Z = Y[::-1]

# # This guy needs some testing
# # R = r"RFRFR-F-RFRFR+F+RFRFRnFn+RFRFR-F-RFRFR+F+RFRFR&F&+RFRFR-F-RFRFR+F+RFRFR<<"  # NOQA
# R = r"R>>FR>>FR>>+F+R>>FR>>FR>>+F+R>>FR>>FRnFn>>-R>>FR>>FR>>-F-R>>FR>>FR>>-F-R>>FR>>FRnFn>>+R>>FR>>FR>>+F+R>>FR>>FR>>+F+R>>FR>>FR"  # NOQA

SUBSTITUTIONS = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "X": X,
    "P": P
    # "P": P,
    # "Y": Y,
    # "Z": Z,
    # "T": T,
    # "U": U,
    # "R": R,
}


def rotz(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def roty(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rotx(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


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


def iterate_lstring(inString: str):
    """
    Iterate an L-String one time

    :param inString:  L-string to iterate
    :return: Iterated L-String
    """
    outString = []
    append = outString.append
    for char in inString:
        if char in SUBSTITUTIONS:
            append(SUBSTITUTIONS[char])
        else:
            append(char)

    return "".join(outString)


def generate_path(lstring: str, n: int = 2, distance: float = 10.0) -> List[np.array]:
    """
    Generate a path from an l-string

    :param lstring: lstring describing path
    :param n: steps on path between forward movements
    :param distance: distance between points forward movements
    :return: list of XYZ points
    """
    axis = np.eye(3)
    pos = [np.array([0, 0, 0])]

    def forward(axis, n=n, distance=distance):
        heading = np.dot(axis, np.array([1, 0, 0]))
        return [(ii + 1) * distance * heading / n for ii in range(0, n)]

    charFunctions = {
        r"+": lambda axis: np.dot(rotu(axis, +np.pi / 2.0), axis),
        r"-": lambda axis: np.dot(rotu(axis, -np.pi / 2.0), axis),
        r"&": lambda axis: np.dot(rotl(axis, +np.pi / 2.0), axis),
        r"n": lambda axis: np.dot(rotl(axis, -np.pi / 2.0), axis),
        r">": lambda axis: np.dot(roth(axis, +np.pi / 2.0), axis),
        r"<": lambda axis: np.dot(roth(axis, -np.pi / 2.0), axis),
        r"|": lambda axis: np.dot(rotu(axis, np.pi), axis),
    }

    for char in lstring:
        if char == r"F":
            lastpos = pos[len(pos) - 1]
            for point in forward(axis):
                pos.append(lastpos + point)
        elif char in charFunctions.keys():
            axis = charFunctions[char](axis)
        else:
            pass

    return pos
