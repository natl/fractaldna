"""
rotations.py
"""
from __future__ import division, unicode_literals, print_function

import numpy as np


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


def eulerMatrix(angx, angy, angz):
    return np.dot(rotz(angz), np.dot(roty(angy), rotx(angx)))


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

    return np.array([psi, theta, phi])
