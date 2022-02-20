import unittest

import matplotlib.pyplot as plt
import numpy as np

from fractaldna.structure_models import random_placements as rp


class TestPrismClass(unittest.TestCase):
    """ """

    def test_prism_properties(self):
        for ii in range(0, 10000):
            point1 = 100 * (np.random.rand(3) - 0.5)
            size1 = 100 * np.random.rand(3)
            ax1 = np.random.rand(3) - 0.5
            r1 = np.random.rand() * 2 * np.pi
            prism = rp.Prism(point1, size1, ax1, r1)
            axis = np.dot(prism.rotation, np.array([1, 0, 0]))
            self.assertAlmostEqual(
                np.dot(axis, prism.axis), 1, 7, "Axis is mal-transformed"
            )
            self.assertAlmostEqual(
                np.dot(prism.norm1, prism.axis),
                0,
                7,
                "Axis and normal are not orthogonal",
            )
            self.assertAlmostEqual(
                np.dot(prism.norm1, prism.norm2), 0, 7, "Normals are not orthogonal"
            )

    def test_contains_generated_points(self):
        for ii in range(0, 10):
            c = (ii + 1) * 2 * (np.random.rand(3) - 0.5)
            s = 10 * np.random.rand(3)
            ax = np.random.rand(3) - 0.5
            r = 2 * np.pi * np.random.rand()
            prism = rp.Prism(c, s, ax, r)
            for jj in range(0, 100):
                self.assertTrue(
                    prism.contains_point(prism.get_point()),
                    "Interior points not found in shape",
                )

    def test_contains_no_excluded_points(self):
        for ii in range(0, 10):
            c = (ii + 1) * 2 * (np.random.rand(3) - 0.5)
            s = 10 * np.random.rand(3)
            ax = np.random.rand(3) - 0.5
            r = 2 * np.pi * np.random.rand()
            prism = rp.Prism(c, s, ax, r)
            for jj in range(0, 100):
                randsign = lambda: 1 if np.random.rand() > 0.5 else -1
                arrsign = np.array([randsign(), randsign(), randsign()])
                p = arrsign * (0.5 * s + 10 * np.random.rand(3))
                p = c + np.dot(prism.rotation, p)
                self.assertFalse(
                    prism.contains_point(p),
                    "Exterior points found in shape:\n"
                    + "position: "
                    + str(p)
                    + "\n"
                    + "iter.: "
                    + str(ii)
                    + ":"
                    + str(jj)
                    + "\n"
                    + "center: "
                    + str(c)
                    + "\n"
                    + "size: "
                    + str(s)
                    + "\n"
                    + "axis: "
                    + str(ax)
                    + "\n"
                    + "rotation: "
                    + str(r),
                )

    def test_overlapping_cylinders(self):
        """ """
        for ii in range(0, 10000):
            point1 = 10 * (np.random.rand(3) - 0.5)
            size1 = np.array([10, 10, 10])  # 50 * np.random.rand(3)
            # size1[0] *= 10
            ax1 = np.random.rand(3)
            ax1 = ax1 / np.linalg.norm(ax1)
            l = (np.random.rand() - 0.5) * size1[0]
            r1 = np.random.rand() * 2 * np.pi
            intersect = point1 + l * ax1
            point2 = 10 * (np.random.rand(3) - 0.5)
            ax2 = intersect - point2
            ax2 = ax2 / np.linalg.norm(ax2)
            l2 = 2 * np.linalg.norm(intersect - point2)
            l2 *= 1 + 2 * np.random.rand()
            size2 = np.array([l2, 1, 1])
            r2 = np.random.rand() * 2 * np.pi

            prism1 = rp.Prism(point1, size1, ax1, r1)
            prism2 = rp.Prism(point2, size2, ax2, r2)

            if not prism1.does_overlap(prism2):
                fig = self.plot_two_prisms(prism1, prism2)  # NOQA
                plt.show()

            self.assertTrue(
                prism1.does_overlap(prism2),
                "Overlapping prisms do not overlap\n"
                + "iter.: "
                + str(ii)
                + "\n"
                + "intersection: "
                + repr(intersect)
                + "\n"
                + "position1: "
                + repr(point1)
                + "\n"
                + "size1: "
                + repr(size1)
                + "\n"
                + "axis1: "
                + repr(ax1)
                + "\n"
                + "position2: "
                + repr(point2)
                + "\n"
                + "size2: "
                + repr(size2)
                + "\n"
                + "axis2: "
                + repr(ax2)
                + "\n\n"
                + "(intsct-p1)/ax1:"
                + repr((intersect - point1) / ax1)
                + "\n"
                + "(intsct-p2)/ax2:"
                + repr((intersect - point2) / ax2),
            )

    @staticmethod
    def plot_two_prisms(prism1, prism2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = []
        y = []
        z = []
        for ii in range(1000):
            p = prism1.get_point()
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        ax.scatter(x, y, z, color="b")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        for (axis, sz) in zip([prism1.axis, prism1.norm1, prism1.norm2], prism1.size):
            a = rp.Arrow3D(
                prism1.center,
                axis * sz + prism1.center,
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="b",
            )
            ax.add_artist(a)

        x = []
        y = []
        z = []
        for ii in range(1000):
            p = prism2.get_point()
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        ax.scatter(x, y, z, color="r")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        for (axis, sz) in zip([prism2.axis, prism2.norm1, prism2.norm2], prism2.size):
            a = rp.Arrow3D(
                prism2.center,
                axis * sz + prism2.center,
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="r",
            )
            ax.add_artist(a)
        return fig
