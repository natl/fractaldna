hilbert3d
---

Routines for generating a fractal that can be 'voxelised'

Try:
```python
import hilbert as h
frac = h.VoxelisedFractal.fromSeed('X', 2)
frac.toPlot(refine=3)
frac.toText()
```

Note that the Peano curve L-strings are incorrect. This is indicated partially
by how they fail the unit tests

**Testing**

Run at the command line:
```
python tests.py
```
