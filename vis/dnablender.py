import bpy
import os
import math
from mathutils import Vector

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
    bpy.data.worlds["World"].light_settings.environment_energy = 0.1
    return None


def make_movie(infile, centre, distance, outfile, clip=100):
    """Make a movie

    make_movie(infile, centre, distance, outfile, clip=100)

    For a given infile, the camera will spin around the geometry
        (around the z-axis), then fly to through along the x-axis to the
        origin, before spinning once more.
    """
    def point_camera(obj_camera, point):
        loc_camera = obj_camera.location
        direction = Vector(point) - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()

    assert os.path.exists(infile), "Could not find infile"
    assert len(centre) == 3, "centre position invalid"
    assert distance > 0, "distance is a positive integer"

    print("Opening {}".format(infile))
    bpy.ops.wm.open_mainfile(filepath=infile)
    cam = bpy.data.cameras.new("RenderCamera")

    cam_ob = bpy.data.objects.new("RenderCamera", cam)
    sceneKey = bpy.data.scenes.keys()[0]
    bpy.data.scenes[sceneKey].objects.link(cam_ob)
    scene = bpy.data.scenes[sceneKey]
    scene.frame_start = 1
    positions = []
    for ii in range(24):
        angle = (ii + 1) * 3.14159 / 12.  # a little less than pi
        positions.append([centre[0] + distance*math.cos(angle),
                          centre[1] + distance*math.sin(angle),
                          centre[2]])

    positions.append((centre[0] + distance, centre[1], centre[2]))

    cam_ob.location = (centre[0] + distance, centre[1], centre[2])
    point_camera(cam_ob, centre)
    the_frame = 1
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)
    for position in positions:
        the_frame = the_frame + 2
        bpy.data.scenes[sceneKey].frame_set(the_frame)
        cam_ob.location = position
        point_camera(cam_ob, centre)
        cam_ob.keyframe_insert(data_path="location", frame=the_frame)
        cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # Pause
    the_frame = the_frame + 6
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # Zoom to center
    the_frame = the_frame + 24
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.location = centre
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # Pause
    the_frame = the_frame + 6
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # Turn around
    the_frame = the_frame + 6
    point_camera(cam_ob, (centre[0], centre[1] - distance, centre[2]))
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)

    the_frame = the_frame + 6
    point_camera(cam_ob, (centre[0] + distance, centre[1], centre[2]))
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)

    # Pause
    the_frame = the_frame + 6
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # Turn again

    the_frame = the_frame + 6
    point_camera(cam_ob, (centre[0], centre[1], centre[2] + distance))
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)

    # Pause
    the_frame = the_frame + 6
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    # zoom back out
    the_frame = the_frame + 24
    bpy.data.scenes[sceneKey].frame_set(the_frame)
    cam_ob.location = [centre[0], centre[1], centre[2] - distance]
    cam_ob.keyframe_insert(data_path="location", frame=the_frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=the_frame)

    bpy.data.cameras["RenderCamera"].clip_end = clip
    bpy.data.scenes[sceneKey].camera = cam_ob
    bpy.data.scenes[sceneKey].render.image_settings.file_format = 'H264'
    bpy.data.scenes[sceneKey].render.fps = 24
    bpy.data.scenes[sceneKey].render.frame_map_old = 100
    bpy.data.scenes[sceneKey].render.frame_map_new = 100
    bpy.data.scenes[sceneKey].render.filepath = outfile
    bpy.data.scenes[sceneKey].render.resolution_x = 400
    bpy.data.scenes[sceneKey].render.resolution_y = 300
    scene.frame_end = the_frame + 1
    print("Rendering {} frames".format(
        int(the_frame)*bpy.data.scenes[sceneKey].render.frame_map_new/
        bpy.data.scenes[sceneKey].render.frame_map_old))
    bpy.ops.render.render(animation=True)
    return None


def make_render(infile, camera_position, camera_rotation, outfile, clip=100):
    """Render a frame

    make_render(infile, camera_position, camera_rotation, outfile, clip=100)

    args:
        infile: blender file to Render
        camera_position: 3-element list for position
        camera_rotation: 3-element list for rotation (degrees)
        outfile: output image render filename
    """
    assert os.path.exists(infile), "Could not find infile"
    assert len(camera_position) == 3, "camera position invalid"
    assert len(camera_position) == 3, "camera direction invalid"
    print("Opening {}".format(infile))
    bpy.ops.wm.open_mainfile(filepath=infile)
    camera_rotation = [3.14159/180.*r for r in camera_rotation]
    cam = bpy.data.cameras.new("RenderCamera")
    cam_ob = bpy.data.objects.new("RenderCamera", cam)
    sceneKey = bpy.data.scenes.keys()[0]
    bpy.data.scenes[sceneKey].objects.link(cam_ob)
    cam_ob.rotation_euler = camera_rotation
    cam_ob.location = camera_position
    bpy.data.cameras["RenderCamera"].clip_end = clip
    bpy.data.scenes[sceneKey].camera = cam_ob
    bpy.data.scenes[sceneKey].render.image_settings.file_format = 'PNG'
    bpy.data.scenes[sceneKey].render.filepath = outfile
    bpy.ops.render.render(write_still=True)
    return None


def assemble_geometry(infile, outfile, units, filepath, placement_dict,
                      ellipse=None):
    """Assemble placement volumes into a blender file.

    assemble_geometry(infile, outfile, units, filepath, placement_dict,
                      **kwargs)

    args:
        infile: input filename (note [1])
        outfile: output filename (must have suffix .blend)
        units: size of each placement volume cube (note [2])
        filepath: directory where input and output file should be
        placement_dict: dictionary, note[3].

    kwargs:
        ellipse: a list of three semi-major axis values that can serve as
                 an xyz mask to restrict placements.
                 Placements are only made inside the ellipse defined by
                 the xyz values given.

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
            kindname is the name the group has in the input placement file
            blendfile should have an absolute path
            groupname is the name of the desired group in the the blendfile
    """
    units = float(units)
    assert outfile[-6:] == ".blend", "outfile param needs .blend suffix"
    if ellipse is not None:
        assert len(ellipse) == 3, "ellipse kwarg contains 3 elements"

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
            pos = [float(ii)*units for ii in ll[2:5]]
            rot = [float(ii) for ii in ll[5:8]]
            valid = True
            if ellipse is not None:
                r = (pos[0]/ellipse[0])**2 + (pos[1]/ellipse[1])**2 +\
                    (pos[2]/ellipse[2])**2
                if r > 1:
                    valid = False
            if valid is True:
                names.append(ll[1])
                positions.append(pos)
                rotations.append(rot)

    for (kind, (path, groupname)) in placement_dict.items():
        with bpy.data.libraries.load(path, link=True) as (data_from, data_to):
            data_to.groups = data_from.groups

    objects = []
    for (ii, (kind, pos, rot)) in enumerate(zip(names, positions, rotations)):
        print("Placement", ii, kind, pos, rot)
        name = str(kind) + str(ii)
        obj = bpy.data.objects.new(name, None)
        obj.dupli_type = "GROUP"
        obj.dupli_group = bpy.data.groups[placement_dict[kind][1]]
        obj.location = pos
        obj.rotation_euler = rot
        objects.append(obj)
        # bpy.ops.object.group_instance_add(group=placement_dict[kind][1],
        #                                   location=pos,
        #                                   rotation=rot)

    for ii, obj in enumerate(objects):
        print("Linking ", ii + 1, " of ", len(objects))
        bpy.context.scene.objects.link(obj)

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
