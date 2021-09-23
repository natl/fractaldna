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

import argparse
import time

import numpy as np
from hilbert3d import hilbert
from mayavi import mlab
from simpledna import dnachain


def make_fractal_movie(seed="X", iter=6):
    """ """
    f = hilbert.VoxelisedFractal.fromSeed(seed, iter)
    f.center_fractal()
    pos = np.array([vox.pos for vox in f.fractal])
    max_pos = np.max(pos, axis=0)
    mask = lambda x: np.sum((x[0:3] / max_pos[0:3]) ** 2) <= 1
    f = f.to_pretty_plot(refine=5, mayavi=True, mask=mask)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig("example/img/fractal_rotate0000.png")
    step = 5
    for ii in range(1, int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        mlab.savefig(f"example/img/fractal_rotate{ii:04d}.png")

    mlab.savefig("example/img/fractal_zoom0000.png")
    nsteps = 200
    for ii in range(nsteps):
        fig.scene.camera.zoom(1.02)
        fig.scene.render()
        mlab.savefig(f"example/img/fractal_zoom{ii:04d}.png")

    for ii in range(nsteps, nsteps + int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        mlab.savefig(f"example/img/fractal_zoom{ii:04d}.png")
    return None


def make_fractal_stills(seed="X"):
    """ """
    f = hilbert.VoxelisedFractal.fromSeed(seed, 1)
    f.center_fractal()
    f.to_pretty_plot(refine=5, mayavi=True)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-1.png")
    mlab.savefig(f"example/stills/{seed}-1.png")

    f = hilbert.VoxelisedFractal.fromSeed(seed, 2)
    f.center_fractal()
    f.to_pretty_plot(refine=5, mayavi=True)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-2.png")
    mlab.savefig(f"example/stills/{seed}-2.png")

    f = hilbert.VoxelisedFractal.fromSeed(seed, 3)
    f.center_fractal()
    f.to_pretty_plot(refine=5, mayavi=True)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-3.png")
    mlab.savefig(f"example/stills/{seed}-3.png")

    f = hilbert.VoxelisedFractal.fromSeed(seed, 4)
    f.center_fractal()
    f.to_pretty_plot(refine=5, mayavi=True)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-4.png")
    mlab.savefig(f"example/stills/{seed}-4.png")

    f = hilbert.VoxelisedFractal.fromSeed(seed, 5)
    f.center_fractal()
    f.to_pretty_plot(refine=5, mayavi=True)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-5.png")
    mlab.savefig(f"example/stills/{seed}-5.png")

    pos = np.array([vox.pos for vox in f.fractal])
    max_pos = np.max(pos, axis=0)
    mask = lambda x: np.sum((x[0:3] / max_pos[0:3]) ** 2) <= 1
    f.to_pretty_plot(refine=5, mayavi=True, mask=mask)
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    mlab.savefig(f"example/stills/{seed}-5-mask.png")
    mlab.savefig(f"example/stills/{seed}-5-mask.png")
    return None


def make_chromatin_movies():
    """ """
    dnachain.MultiSolenoidVolume(voxelheight=1500, separation=400).to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_multi_straight0000.png")
    step = 5
    for ii in range(1, int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = f"example/img/chromatin_multi_straight{ii:04d}.png"
        mlab.savefig(fname)

    dnachain.MultiSolenoidVolume(
        voxelheight=1500, separation=400, turn=True
    ).to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_multi_turn0000.png")
    step = 5
    for ii in range(1, int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = f"example/img/chromatin_multi_turn{ii:04d}.png"
        mlab.savefig(fname)

    dnachain.TurnedSolenoid().to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_single_turn0000.png")
    step = 5
    for ii in range(1, int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = f"example/img/chromatin_single_turn{ii:04d}.png"
        mlab.savefig(fname)

    dnachain.Solenoid().to_line_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.savefig("example/img/chromatin_single_straight0000.png")
    step = 5
    for ii in range(1, int(720 / step)):
        fig.scene.camera.azimuth(step)
        fig.scene.render()
        fname = f"example/img/chromatin_single_straight{ii:04d}.png"
        mlab.savefig(fname)
    return None


def make_still_dna_pieces():
    """ """
    dnachain.Solenoid().to_line_plot(size=[1024, 768])
    fig = mlab.gcf()
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90)
    mlab.savefig("example/stills/straight_line_plot.png")
    mlab.savefig("example/stills/straight_line_plot.png")

    dnachain.Solenoid().to_strand_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90)
    mlab.savefig("example/stills/straight_strand_plot.png")
    mlab.savefig("example/stills/straight_strand_plot.png")

    dnachain.TurnedSolenoid().to_line_plot(size=[1024, 768])
    fig = mlab.gcf()
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90, azimuth=-90)
    mlab.savefig("example/stills/turn_line_plot.png")
    mlab.savefig("example/stills/turn_line_plot.png")

    dnachain.TurnedSolenoid().to_strand_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90, azimuth=-90)
    mlab.savefig("example/stills/turn_strand_plot.png")
    mlab.savefig("example/stills/turn_strand_plot.png")

    dnachain.TurnedSolenoid(twist=True).to_line_plot(size=[1024, 768])
    fig = mlab.gcf()
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90, azimuth=-90)
    mlab.savefig("example/stills/turn_twist_line_plot.png")
    mlab.savefig("example/stills/turn_twist_line_plot.png")

    dnachain.TurnedSolenoid(twist=True).to_strand_plot()
    fig = mlab.gcf()
    fig.scene.set_size([1024, 768])
    fig.scene.render()
    time.sleep(1)
    mlab.view(elevation=90, azimuth=-90)
    mlab.savefig("example/stills/turn_twist_strand_plot.png")
    mlab.savefig("example/stills/turn_twist_strand_plot.png")

    d = dnachain.Solenoid()
    print(f"Solenoid has {len(d.basepairs)} bp, {len(d.histones)} histones")
    d = dnachain.TurnedSolenoid()
    print(
        "Turned Solenoid has {} bp, {} histones".format(
            len(d.basepairs), len(d.histones)
        )
    )
    d = dnachain.TurnedSolenoid(twist=True)
    print(
        "Turned Twisted Solenoid has {} bp, {} histones".format(
            len(d.basepairs), len(d.histones)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command Line examples of "
        + "how this package works, with movie and text file generation"
    )  # NOQA
    parser.add_argument("option")
    parser.add_argument("--onscreen", action="store_true")
    args = parser.parse_args()
    if not args.onscreen:
        mlab.options.offscreen = True
    if args.option == "chromatin_movie":
        make_chromatin_movies()
    elif args.option == "images":
        make_still_dna_pieces()
        make_fractal_stills()
    elif args.option == "fractal_movie":
        make_fractal_movie()
