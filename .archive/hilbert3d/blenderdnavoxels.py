import bpy
import numpy as np

p = np.loadtxt("/Users/nlampe/OneDrive/Doctorate/hilbert3d/dna.txt", delimiter=",")

height = max(p[:, 6]) - p[-1, 6]

voxelsize = 150.0  # 500.  # angstroms


def roty(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def placeAtom(size, r, g, b, x, y, z, groupname=""):
    """ """
    print(f"Size of Atom\n{size}")

    print("Color of Atom")
    print(f"red = {r}, green = {g}, blue = {b}")

    print("Atom Coordinates")
    print(f"({x}, {y}, {z})\n")

    # Make and locate Atom
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=8,
        size=size,
        view_align=False,
        enter_editmode=False,
        location=(x, y, z),
        rotation=(0, 0, 0),
    )
    if groupname:
        bpy.ops.object.group_link(group=groupname)

    # Make Material
    mat = bpy.data.materials.new("Color01")
    mat.diffuse_color = (r, g, b)
    mat.diffuse_shader = "LAMBERT"
    mat.diffuse_intensity = 1.0
    mat.specular_color = (1, 1, 1)
    mat.specular_shader = "COOKTORR"
    mat.specular_intensity = 0.5
    mat.alpha = 1
    mat.ambient = 1
    bpy.context.object.data.materials.append(mat)

    return None


def drawLinear():
    """ """
    reps = round(voxelsize / height)
    dna = p.copy()
    for ii in range(1, int(reps)):
        t = p.copy()
        t[:, 6] += ii * height
        dna = np.concatenate((dna, t), axis=0)
    scaling = voxelsize / (reps * height)
    dna[:, 6] *= scaling

    bpy.ops.group.create(name="dnastraight")
    for atom in dna:
        [size, r, g, b, x, y, z] = atom
        placeAtom(size, r, g, b, x, y, z, groupname="dnastraight")

    return None


def drawCurved():
    """ """
    length = voxelsize * np.pi / 4.0  # circumference to be covered by DNA
    reps = round(length / height)
    scaling = length / (reps * height)
    dna = p.copy()
    for ii in range(1, int(reps)):
        t = p.copy()
        t[:, 6] += ii * height
        dna = np.concatenate((dna, t), axis=0)

    dna[:, 6] *= scaling

    # Turn the DNA now
    zmax = length
    radius = voxelsize / 2.0
    for atom in dna:
        atomx = atom[4]
        atomy = atom[5]
        atomz = atom[6]

        # Translation of the frame - new center position
        theta = np.pi / 2.0 * atomz / zmax
        neworigin = np.array(
            [radius * np.cos(theta) - radius, 0.0, radius * np.sin(theta)]
        )

        # rotation of the frame
        oldframe = np.array([atomx, atomy, 0])
        yrotation = -np.pi / 2.0 * atomz / zmax

        newframe = np.dot(roty(yrotation), oldframe)
        atom[4] = neworigin[0] + newframe[0]
        atom[5] = neworigin[1] + newframe[1]
        atom[6] = neworigin[2] + newframe[2]

    bpy.ops.group.create(name="dnaturn")
    for atom in dna:
        [size, r, g, b, x, y, z] = atom
        placeAtom(size, r, g, b, x, y, z, groupname="dnaturn")

    return None


if __name__ == "__main__":
    drawCurved()
