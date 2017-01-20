import bpy
import os
import pdb

COLOURS = {"PHOSPHATE": (255, 255, 0),
           "SUGAR": (255, 0, 0),
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
    mat.ambient = 0.05
    MATERIALS[k] = mat


def do_world():
    """Make the world look nice
    """
    bpy.data.worlds["World"].use_sky_blend = 1
    bpy.data.worlds["World"].horizon_color = (0.27, 1, 1)
    bpy.data.worlds["World"].zenith_color = (1, 1, 1)
    bpy.data.worlds["World"].ambient_color = (1, 1, 1)
    bpy.data.worlds["World"].light_settings.use_environment_light = 1
    return None


def make_movie(infile, camera_positions, camera_directions):
    """
    """
    return None


def assemble_geometry(infile, outfile, units, filepath, placement_dict):
    """Assemble placement volumes into a blender file.

    assemble_geometry(infile, outfile, units, filepath, placement_dict)

    args:
        infile: input filename (note [1])
        outfile: output filename (must have suffix .blend)
        units: size of each placement volume cube (note [2])
        filepath: directory where input and output file should be
        placement_dict: dictionary, note[3].

    note:
        [1]
        structure of input file (whitespace separated columns)
            IDX KIND POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI

        [2]
        The file instructing blender where to place volumes may be parametrised
        so it can work with a variety of placement groups. This parameter
        is a multiplicative factor that multiplies the POS_X, POS_Y and POS_Z
        columns.
        For example, the input file may provide placements only in integer
        units in a grid, and then the multiplicative value here can be used
        to specify the size of each placement, ex: 50 units (units are relative
        to whatever units are in the placement files)

        [3]
        placement_dict contains key/value pairs for:
            (kindname, (blendfile, groupname))
            blendfile should have an absolute path
            groupname is the name of the desired group in the the blendfile
    """
    units = float(units)
    assert outfile[-6:] == ".blend", "outfile param needs .blend suffix"

    bpy.ops.wm.read_homefile()
    for t in ["MESH", "SURFACE", "CURVE", "META", "FONT", "CAMERA", "LAMP"]:
        bpy.ops.object.select_by_type(type=t)
        bpy.ops.object.delete()

    # set up file
    do_world()

    # read_in
    infile  = open(os.path.join(filepath, infile), "r")
    names = []
    positions = []
    rotations = []
    for line in infile:
        if line[0] != "#":
            ll = line.split()
            names.append(ll[1])
            pos = [float(ii)*units for ii in ll[2:5]]
            rot = [float(ii) for ii in ll[5:8]]
            positions.append(pos)
            rotations.append(rot)

    for (kind, (path, groupname)) in placement_dict.items():
        with bpy.data.libraries.load(path, link=True) as (data_from, data_to):
            data_to.groups = data_from.groups

    for (ii, (kind, pos, rot)) in enumerate(zip(names, positions, rotations)):
        print(ii, kind, pos, rot)
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
        [1]
        structure of input file:
        col00:    NAME
        col01:    SHAPE
        col02:    CHAIN_ID
        col03:    STRAND_ID
        col04:    BP_INDEX
        col05-07: SIZE_X SIZE_Y SIZE_Z
        col08-10: POS_X POS_Y POS_Z
        col11-13: ROT_X ROT_Y ROT_Z

        [2]
        A default groupname is created for each placement, corresponding
        to the value of outfile, without the .blend suffix
    """
    assert outfile[-6:] == ".blend", "outfile param needs .blend suffix"

    bpy.ops.wm.read_homefile()
    for t in ["MESH", "SURFACE", "CURVE", "META", "FONT", "CAMERA", "LAMP"]:
        bpy.ops.object.select_by_type(type=t)
        bpy.ops.object.delete()

    do_world()

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
        mat.ambient = 0.05
        ob.data.materials.append(mat)

    outfile = os.path.join(filepath, outfile)
    bpy.ops.wm.save_as_mainfile(filepath=outfile)
    return None
