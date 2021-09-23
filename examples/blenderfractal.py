# To run:
# filename="/Users/nlampe/OneDrive/Doctorate/bacdna/hilbert3d/blenderfractal.py"
# exec(compile(open(filename).read(), filename, 'exec'))
import sys

import bpy

sys.path.append("/Users/nlampe/OneDrive/Doctorate/bacdna/hilbert3d/")
import hilbert as h


# Make fractal
def makeFractal(n=1, s="X"):
    for ii in range(n):
        s = h.iterate(s)
    frac = h.VoxelisedFractal.fromLString(s, distance=70)
    return frac


# Use the voxelised fractal to seed a new blender file.
def blenderifyFractal(vf):
    """ """
    imgdir = "/Users/nlampe/OneDrive/Doctorate/fractaldna/hilbert3d/img/"
    filepaths = [
        imgdir + "dnaturn.blend",
        imgdir + "dnastraight.blend",
        imgdir + "dnaturntwist.blend",
    ]

    for filepath in filepaths:
        with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
            data_to.groups = data_from.groups

    mdict = {
        h.Voxel.types["turn"]: "dnaturn",
        h.Voxel.types["turntwist"]: "dnaturntwist",
        h.Voxel.types["straight"]: "dnastraight",
    }

    ii = 0
    for voxel in vf.fractal:
        shape = mdict[voxel.type]
        bpy.ops.object.group_instance_add(
            group=shape,
            location=tuple(voxel.pos),
            rotation=(voxel.psi, voxel.theta, voxel.phi),
        )
        print(voxel.psi, voxel.theta, voxel.phi)
        # scn = bpy.context.scene
        # scn.objects.link(ob)
        ii = ii + 1

    return None


def main():
    vf = makeFractal()
    blenderifyFractal(vf)

    return None


if __name__ == "__main__":
    main()
