SimpleDNA
===
Little scripts for generate strings of DNA
---

Here are a few scripts I use for generating straight, turned and twisted DNA chains based on a geometrical description of DNA.

The methods you will need to call to generate straight our curved DNA segments are found in `dnachain.py`, while the folder `utils` contains a collection of classes for building molecule shapes, bases and sequences.

Some unit testing is handled in by calling `python tests.py`.

## Plotting Solenoidal DNA ##

Solenoidal DNA can be generated using the dnachain module. Plots of solenoidal DNA often require mayavi2 for visualisation

Install this with `pip install mayavi` or `apt-get install python-mayavi2`

Then try

```.py
import danchain as d
sol = d.Solenoid()
sol.to_strand_plot()
sol.to_line_plot()

sol = d.TurnedSolenoid()
sol.to_strand_plot()
sol.to_line_plot()

with open("solenoidal.txt", "w") as f:
    f.write(sol.to_text())

```

