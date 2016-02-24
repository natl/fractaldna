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
