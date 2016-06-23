"""
Generate a sequence of a GATC values of a certain length
"""
from __future__ import division, print_function, unicode_literals
from . import BP_SEPARATION
import numpy as np

def get_dna_string(length, separation=BP_SEPARATION):
    """Return a string of letters representing a DNA strand of a set length
    get_dna_string(length, separation)

    args:
        length: length of strand to construct (default units angstrom)

    kwargs:
        separation: specified separation between base pairs, imported from
                    utils folder

    notes:
        Both length and separation should have the same units, the default
        separation is set in angstroms
    """
    candidates = ["G", "A", "T", "C"]
    strand = []
    for ii in range(0, int(length//separation)):
        strand.append(np.random.choice(candidates))

    return "".join(strand)
