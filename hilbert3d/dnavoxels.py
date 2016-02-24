import numpy as np
import bpy

p = np.loadtxt('/Users/nlampe/OneDrive/Doctorate/hilbert3d/dna.txt',
               delimiter=',')

height = max(p[:, 6]) - p[-1, 6]

voxelsize = 70.  # 500.  # angstroms


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


def placeAtom(size, r, g, b, x, y, z, groupname=""):
    """
    """
    print('Size of Atom\n{0}'.format(size))

    print('Color of Atom')
    print("red = {0}, green = {1}, blue = {2}".format(r, g, b))

    print('Atom Coordinates')
    print('({0}, {1}, {2})\n'.format(x, y, z))

    # Make and locate Atom
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8,
                                         size=size,
                                         view_align=False,
                                         enter_editmode=False,
                                         location=(x, y, z),
                                         rotation=(0, 0, 0))
    if groupname:
        bpy.ops.object.group_link(group=groupname)

    # Make Material
    mat = bpy.data.materials.new('Color01')
    mat.diffuse_color = (r, g, b)
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = (1, 1, 1)
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = (1)
    mat.ambient = 1
    bpy.context.object.data.materials.append(mat)

    return None


def drawLinear():
    """
    """
    reps = round(voxelsize/height)
    dna = p.copy()
    for ii in range(1, int(reps)):
        t = p.copy()
        t[:, 6] += ii*height
        dna = np.concatenate((dna, t), axis=0)
    scaling = voxelsize/(reps*height)
    dna[:, 6] *= scaling

    dna[:, 6] -= voxelsize/2.
    bpy.ops.group.create(name="dnastraight")
    for atom in dna:
        [size, r, g, b, x, y, z] = atom
        placeAtom(size, r, g, b, x, y, z, groupname="dnastraight")

    return None


def drawTurn():
    """
    """
    length = voxelsize*np.pi/4.  # circumference to be covered by DNA
    reps = round(length/height)
    scaling = length/(reps*height)
    dna = p.copy()
    for ii in range(1, int(reps)):
        t = p.copy()
        t[:, 6] += ii*height
        dna = np.concatenate((dna, t), axis=0)

    dna[:, 6] *= scaling


    # Turn the DNA now
    zmax = length
    radius = voxelsize/2.
    for atom in dna:
        atomx = atom[4]
        atomy = atom[5]
        atomz = atom[6]

        # Translation of the frame - new center position
        theta = np.pi/2.*atomz/zmax
        neworigin = np.array([radius*(1 - np.cos(theta)),
                              0.,
                              radius*np.sin(theta)])

        # rotation of the frame
        oldframe = np.array([atomx, atomy, 0])
        yrotation = np.pi/2.*atomz/zmax

        newframe = np.dot(roty(yrotation), oldframe)
        atom[4] = neworigin[0] + newframe[0]
        atom[5] = neworigin[1] + newframe[1]
        atom[6] = neworigin[2] + newframe[2]

    dna[:, 6] -= voxelsize/2.
    bpy.ops.group.create(name="dnaturn")
    for atom in dna:
        [size, r, g, b, x, y, z] = atom
        placeAtom(size, r, g, b, x, y, z, groupname="dnaturn")

    return dna.copy()


def drawTurnTwist():
    """
    """
    length = voxelsize*np.pi/4.  # circumference to be covered by DNA
    reps = round(length/height)
    scaling = length/(reps*height)
    dna = p.copy()
    for ii in range(1, int(reps)):
        t = p.copy()
        t[:, 6] += ii*height
        dna = np.concatenate((dna, t), axis=0)

    dna[:, 6] *= scaling

    # Turn the DNA now
    zmax = length
    radius = voxelsize/2.
    for atom in dna:
        atomx = atom[4]
        atomy = atom[5]
        atomz = atom[6]

        # Translation of the frame - new center position
        theta = np.pi/2.*atomz/zmax
        neworigin = np.array([radius*(1 - np.cos(theta)),
                              0.,
                              radius*np.sin(theta)])

        # rotation of the frame, with a twist
        oldframe = np.array([atomx, atomy, 0])
        yrotation = np.pi/2.*atomz/zmax
        xrotation = np.pi/2.*atomz/zmax

        newframe = np.dot(rotx(xrotation), np.dot(roty(yrotation), oldframe))

        atom[4] = neworigin[0] + newframe[0]
        atom[5] = neworigin[1] + newframe[1]
        atom[6] = neworigin[2] + newframe[2]

    dna[:, 6] -= voxelsize/2.
    bpy.ops.group.create(name="dnaturntwist")
    for atom in dna:
        [size, r, g, b, x, y, z] = atom
        placeAtom(size, r, g, b, x, y, z, groupname="dnaturntwist")

    return dna.copy()

if __name__ == "__main__":
    path = "/Users/nlampe/OneDrive/Doctorate/hilbert3d/"
    # drawTurnTwist()
    # bpy.ops.wm.save_as_mainfile(filepath=path+"img/dnaturntwist.blend")

    drawLinear()
    bpy.ops.wm.save_as_mainfile(filepath=path+"img/dnastraight.blend")

    # drawTurn()
    # bpy.ops.wm.save_as_mainfile(filepath=path+"img/dnaturn.blend")

    # filename = "/Users/nlampe/OneDrive/Doctorate/hilbert3d/dnavoxels.py"
    # exec(compile(open(filename).read(), filename, 'exec'))
