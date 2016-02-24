from __future__ import division, print_function, unicode_literals

import unittest
import dnachain
import basepair

import numpy as np


class TestDNACreation(unittest.TestCase):
    """
    """
    def test_dnaLength(self):
        """
        """
        genome = "GCGCGCGCTATATATATCTCTCACACA"
        x = dnachain.DNAChain(genome)
        self.assertEqual(len(x.basepairs), len(genome),
                         "Genome is incorrect length")
        nMolecules = sum([len(bp.molecules) for bp in x.basepairs])
        self.assertEqual(nMolecules, len(genome)*6,
                         "Incorrect number of molecules")


class TestBasePairs(unittest.TestCase):
    """
    """
    def test_basepair_creation_rotation_translation(self):
        translation = np.array([1, 2, 3])
        rotation = np.array([0.5, 0.25, 1])*np.pi
        bp1 = basepair.BasePair("G")
        bp2 = basepair.BasePair("G", position=translation,
                                rotation=rotation)
        bp1.rotate(rotation)
        bp1.translate(translation)
        for name in basepair.BasePair.moleculeNames:
            self.assertAlmostEqual(
                np.sum(bp1[name].position - bp2[name].position), 0)
            self.assertAlmostEqual(
                np.sum(bp1[name].rotation - bp2[name].rotation), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
