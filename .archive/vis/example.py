#!/bin/python3
# example.py
# Example python file for blender renders of DNA geometries
# Requires Blender to be installed, this script is designed to be run
# without the GUI (blender option -p).
#
# ie. try blender -b -P example.py

import os
import sys

the_path = os.path.abspath(os.path.curdir)

sys.path.append(the_path)

import dnablender

if __name__ == "__main__":
    dnablender.placement_volume(
        "4strands_50nm_straight.txt", "straight.blend", the_path
    )
    dnablender.placement_volume("4strands_50nm_turn.txt", "turn.blend", the_path)
    dnablender.assemble_geometry(
        "hilbert2.txt",
        "hilbert2.blend",
        500,
        the_path,
        {
            "straight": (os.path.join(the_path, "straight.blend"), "straight"),
            "turn": (os.path.join(the_path, "turn.blend"), "turn"),
            "turntwist": (os.path.join(the_path, "turn.blend"), "turn"),
        },
    )
    dnablender.make_render(
        os.path.join(the_path, "hilbert2.blend"),
        [750, 750, -3000],
        [-180, 0, 0],
        os.path.join(the_path, "hilbert2.png"),
        clip=15000,
    )
    dnablender.make_movie(
        os.path.join(the_path, "hilbert2.blend"),
        [750, 750, 750],
        2500,
        os.path.join(the_path, "hilbert2.mp4"),
        clip=10000,
    )
