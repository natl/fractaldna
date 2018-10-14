#!/bin/python
"""
mayavi_example.py
"""

from mayavi import mlab
from hilbert3d import hilbert
import numpy as np
import argparse

def make_fractal_movie(seed="X", iter=5):
    """
    """
    f = hilbert.VoxelisedFractal.fromSeed(seed, iter)
    f.center_fractal()
    pos = np.array([vox.pos for vox in f.fractal])
    max_pos = np.max(pos, axis=0)
    mask = lambda x: np.sum((x[0:3]/max_pos[0:3])**2) <= 1
    f = f.to_pretty_plot(refine=5, mayavi=True, mask=mask)
    fig = mlab.gcf()
    mlab.savefig("example/img/fractal0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        mlab.savefig("example/img/fractal{:04d}.png".format(ii))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line examples of " +
        "how this package works, with movie and text file generation")  # NOQA
    mlab.options.offscreen = True
    make_fractal_movie()
