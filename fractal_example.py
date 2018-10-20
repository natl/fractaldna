#!/bin/python
"""
mayavi_example.py

How to turn images into a video:
ffmpeg -f image2 -r 10 -i example/img/fractal%04d.png -q:v 4 -c:v h264 \
    -pix_fmt yuv420p example/img/fractal.mp4

ls *.png | sed s/[0-9]*.png//g | uniq

for f in `ls *.png | sed s/[0-9]*.png//g | uniq`
    do ffmpeg -y -f image2 -r 10 -i $f%04d.png -q:v 4 -c:v h264 \
        -pix_fmt yuv420p $f.mp4
done
"""

from mayavi import mlab
from hilbert3d import hilbert
from simpledna import dnachain
import numpy as np
import argparse
import time


def make_fractal_movie(seed="X", iter=6):
    """
    """
    f = hilbert.VoxelisedFractal.fromSeed(seed, iter)
    f.center_fractal()
    pos = np.array([vox.pos for vox in f.fractal])
    max_pos = np.max(pos, axis=0)
    mask = lambda x: np.sum((x[0:3]/max_pos[0:3])**2) <= 1
    f = f.to_pretty_plot(refine=5, mayavi=True, mask=mask)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig("example/img/fractal_rotate0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        mlab.savefig("example/img/fractal_rotate{:04d}.png".format(ii))

    mlab.savefig("example/img/fractal_zoom0000.png")
    nsteps = 200
    for ii in range(nsteps):
        fig.scene.camera.zoom(1.02)
        fig.scene.render()
        mlab.savefig("example/img/fractal_zoom{:04d}.png".format(ii))

    for ii in range(nsteps, nsteps + int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        mlab.savefig("example/img/fractal_zoom{:04d}.png".format(ii))
    return None


def make_chromatin_movies():
    """
    """
    dnachain.MultiSolenoidVolume(voxelheight=1500, separation=400)\
        .to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_multi_straight0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = "example/img/chromatin_multi_straight{:04d}.png".format(ii)
        mlab.savefig(fname)

    dnachain.MultiSolenoidVolume(voxelheight=1500, separation=400, turn=True)\
        .to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_multi_turn0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = "example/img/chromatin_multi_turn{:04d}.png".format(ii)
        mlab.savefig(fname)

    dnachain.TurnedSolenoid().to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_single_turn0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = "example/img/chromatin_single_turn{:04d}.png".format(ii)
        mlab.savefig(fname)

    dnachain.Solenoid().to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_single_straight0000.png")
    step = 5
    for ii in range(1, int(720/step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = "example/img/chromatin_single_straight{:04d}.png".format(ii)
        mlab.savefig(fname)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line examples of " +
        "how this package works, with movie and text file generation")  # NOQA
    # mlab.options.offscreen = True
    # make_fractal_movie()
    make_chromatin_movies()
