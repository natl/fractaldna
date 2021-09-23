hilbert3d
---

Routines for generating a fractal that can be 'voxelised'

Try:
```python
import hilbert as h
frac = h.VoxelisedFractal.fromSeed('X', 2)
frac.toPlot(refine=3)
frac.toText()

# And then try other seeds
# ABCDX -> seeds for Hilbert-like curves
# P -> Peano-like curve (still in testing)
# Adding F to join letters often decreases the density
fig = h.VoxelisedFractal.fromSeed('A', 2).toPlot(refine=3)
fig = h.VoxelisedFractal.fromSeed('AFFC', 2).toPlot(refine=3)
fig = h.VoxelisedFractal.fromSeed('P', 2).toPlot(refine=3)
```

Note that the Peano curve L-system strings are incorrect. This is indicated
partially by how they fail the unit tests

**Testing**

Run at the command line:
```
python tests.py
```
