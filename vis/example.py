#!/bin/python3
# example.py
# Example python file for blender renders of DNA geometries
# Requires Blender to be installed, this script is designed to be run
# without the GUI (blender option -p).
#
# ie. try blender -b -P example.py

import os
import dnablender

the_path = os.path.abspath(os.path.curdir)

if __name__ == "__main__":
    dnablender.placement_volume("4strands_50nm_straight.txt",
                                "straight.blend", the_path)
    dnablender.placement_volume("4strands_50nm_turn.txt",
                                "turn.blend", the_path)
    dnablender.assemble_geometry(
        "hilbert2.txt", "hilbert2.blend", 500, the_path,
        {"straight": (os.path.join(the_path, "straight.blend"), "straight"),
         "turn": (os.path.join(the_path, "turn.blend"), "turn"),
         "turntwist": (os.path.join(the_path, "turn.blend"), "turn")})
