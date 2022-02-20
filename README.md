FractalDNA
===
Python routines for generating geometric models of DNA
---

*FractalDNA is being converted to a package, it is under active developmemt*

<div align="center">

[![Build status](https://github.com/natl/fractaldna/workflows/build/badge.svg?branch=master&event=push)](https://github.com/natl/fractaldna/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/fractaldna.svg)](https://pypi.org/project/fractaldna/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/natl/fractaldna/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/natl/fractaldna/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/natl/fractaldna/releases)
[![License](https://img.shields.io/github/license/natl/fractaldna)](https://github.com/natl/fractaldna/blob/master/LICENSE)

</div>

FractalDNA is a Python package to make DNA geometries that can be joined together like
jigsaw puzzles. Both simple, sections of DNA and Solenoidal DNA can be built. This
module was built to enable DNA-level simulations to be run in [Geant4-DNA](http://geant4-dna.in2p3.fr/), part of the
[Geant4](geant4.cern.ch/) project.

Structure models define the large scale structure of DNA,
seeded from fractals. An example seeding fractal is below:

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/natl/fractaldna/master/docs/source/images/fractal-path.svg" alt="A 3-D iterated Hilbert Curve">
</p>

DNA Models provide straight and curved segments that can come together to
make DNA for use in simulations.

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/natl/fractaldna/master/docs/source/images/single_solenoid_line_plot.jpg" alt="A straight solenoidal DNA segment">
</p>

Project documentation is available [here](http://natl.github.io/fractaldna/) alongside [notebook examples](http://natl.github.io/fractaldna/examples.html)

## ⚙️ Installation

Install FractalDNA with `pip`

```bash
pip install fractaldna
```

or install with `Poetry`

```bash
poetry add fractaldna
```

## 🧬 Make some DNA

```py
from fractaldna.dna_models import dnachain as dna

# Make a DNA Chain 40 base pairs long (repeating GATC).
chain = dna.DNAChain("GTAC" * 10)

# Export it to a DataFrame for use in another program
df = chain.to_frame()
```

For more, check out the [notebook examples](http://natl.github.io/fractaldna/examples.html) in the project documentation.

## 🛡 License

[![License](https://img.shields.io/github/license/natl/fractaldna)](https://github.com/natl/fractaldna/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/natl/fractaldna/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{fractaldna,
  author = {Nathanael Lampe},
  title = {FractalDNA},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/natl/fractaldna}}
}
```

## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)