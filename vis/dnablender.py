import bpy
import os
import pdb

COLOURS = {"TRIPHOSPHATE": (255, 255, 0),
           "DNASUGAR": (255, 0, 0),
           "GUANINE": (0, 255, 0),
           "ADENINE": (0, 0, 255),
           "CYTOSINE": (0, 255, 255),
           "THYMINE": (255, 0, 255)}


MATERIALS = {}
for (k, v) in COLOURS.items():
    # Make Material
    mat = bpy.data.materials.new(k)
    mat.diffuse_color = v
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = v
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = (1)
    mat.ambient = 1
    MATERIALS[k] = mat


def assemble_geometry(infile, outfile, filepath, placement_dict):
    """Assemble placement volumes into a blender file.

    args:
        infile
        outfile
        filepath
        placement_dict

    note:
        structure of input file (whitespace separated columns)
            IDX KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI

        placement_dict contains key/value pairs for:
            (kindname, (blendfile, groupname))
            blendfile should have an absolute path
            groupname is the name of the desired group in the the blendfile
    """
    assert outfile[-6:] == ".blend", "outfile param needs .blend suffix"

    bpy.ops.wm.read_homefile()
    for t in ["MESH", "SURFACE", "CURVE", "META", "FONT", "CAMERA", "LAMP"]:
        bpy.ops.object.select_by_type(type=t)
        bpy.ops.object.delete()

    # set up file
    bpy.data.worlds["World"].use_sky_blend = True
    bpy.data.worlds["World"].zenith_color = (0., 1., 1.)
    bpy.data.worlds["World"].horizon_color = (1., 1., 1.)
    bpy.ops.objects.lamp_add(type="HEMI", location=(0., 0., 1000.))

    # read_in
    infile  = open(os.path.join(filepath, infile), "r")
    names = []
    positions = []
    rotations = []
    for line in infile:
        if line[0] != "#":
            ll = line.split()
            names.append(ll[1])
            pos = [float(ii) for ii in ll[2:5]]
            rot = [float(ii) for ii in ll[6:9]]
            positions.append(pos)
            rotations.append(rot)

    for (kind, (path, groupname)) in placement_dict:
        with bpy.data.libraries.load(path, link=True) as (data_from, data_to):
            data_to.groups = data_from.groups

    for (ii, (kind, pos, rot)) in enumerate(zip(names, positions, rotations)):
        bpy.ops.object.group_instance_add(group=placement_dict[kind][1],
                                          location=pos,
                                          rotation=rot)

    outfile = os.path.join(filepath, outfile)
    bpy.ops.wm.save_as_mainfile(filepath=outfile)
    return None


def placement_volume(infile, outfile, filepath):
    """Load a placement volume into blender and save it

    args:
        infile: string, input file name (see below for structure)
        outfile: output file (with .blend extension)
        filepath: directory where both infile and outfile should/will be.

    note:
        structure of input file:
        col00:    NAME
        col01:    SHAPE
        col02:    CHAIN_ID
        col03:    STRAND_ID
        col04:    BP_INDEX
        col05-07: SIZE_X SIZE_Y SIZE_Z
        col08-10: POS_X POS_Y POS_Z
        col11-13: ROT_X ROT_Y ROT_Z
    """
    assert outfile[-6:] == ".blend", "outfile param needs .blend suffix"

    bpy.ops.wm.read_homefile()
    for t in ["MESH", "SURFACE", "CURVE", "META", "FONT", "CAMERA", "LAMP"]:
        bpy.ops.object.select_by_type(type=t)
        bpy.ops.object.delete()

    infile  = open(os.path.join(filepath, infile), "r")
    names = []
    sizes = []
    positions = []
    rotations = []
    for line in infile:
        if line[0] != "#":
            ll = line.split()
            names.append(ll[0])
            sz = [float(ii) for ii in ll[5:8]]
            pos = [float(ii) for ii in ll[8:11]]
            rot = [float(ii) for ii in ll[11:14]]

            sizes.append(sz)
            positions.append(pos)
            rotations.append(rot)

    groupname = outfile.replace(".blend", "")
    bpy.ops.group.create(name=groupname)
    for (ii, (name, sz, pos, rot)) in\
            enumerate(zip(names, sizes, positions, rotations)):
        if (ii//100)*100 == ii:
            print("{0}/{1}".format(ii, len(names)))
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8,
                                             size=1,
                                             view_align=False,
                                             enter_editmode=False,
                                             location=pos,
                                             rotation=rot)
        bpy.ops.object.group_link(group=groupname)

        ob = bpy.context.object
        ob.scale = sz
        try:
            color = COLOURS[name.upper()]
        except KeyError:
            print("Material not defined for {}".format(name.upper()))
            color = (100, 100, 100)

        mat = bpy.data.materials.new(k)
        mat.diffuse_color = color
        mat.diffuse_shader = 'LAMBERT'
        mat.diffuse_intensity = 1.0
        mat.specular_color = (1, 1, 1)
        mat.specular_shader = 'COOKTORR'
        mat.specular_intensity = 0.5
        mat.alpha = (1)
        mat.ambient = 1
        ob.data.materials.append(mat)

    outfile = os.path.join(filepath, outfile)
    bpy.ops.wm.save_as_mainfile(filepath=outfile)
    return None
