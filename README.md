fractaldna
===
Python routines for generating simple models of DNA
---

*FractalDNA is being converted to a package, it is under active developmemt*

<div align="center">

[![Build status](https://github.com/fractaldna/fractaldna/workflows/build/badge.svg?branch=master&event=push)](https://github.com/fractaldna/fractaldna/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/fractaldna.svg)](https://pypi.org/project/fractaldna/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/fractaldna/fractaldna/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/fractaldna/fractaldna/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/fractaldna/fractaldna/releases)
[![License](https://img.shields.io/github/license/fractaldna/fractaldna)](https://github.com/fractaldna/fractaldna/blob/master/LICENSE)

</div>

This repository is an offshoot of my thesis work, where I simulate the impact
of ionising radiation on DNA. For this, I need to model and visualise very
large DNA structures

Modelling DNA geometries computationally can be done very crudely based on
a few DNA motifs and a fractal geometry. It provides a memory efficient way of
ensuring that an appropriate density of DNA is placed in a sample volume. The
idea is to use a fractal as a seed for a collection of turned and straight
geometries, and then place repeating turned and straight DNA segments inside
these geometries.

Here you can see the idea being applied to the first few iterations of a Hilbert
curve.

![Fractal DNA](https://cloud.githubusercontent.com/assets/2887977/22364141/936da1ee-e46f-11e6-9c56-ee4e0dcb8d0f.png)

The project is divided into three sections, each with their own Readme:
* `hilbert3d` provides routines for generating 3D fractals from L-systems.
* `simpledna` has some routines for generating simple turned and straight
DNA models.
* `vis` contains some Python routines that work in Blender to visualise the
whole DNA structure.

This project is currently in a beta form, I'm working on documentation and
the ability to generate videos of DNA procedurally in Blender from Python
scripts. Though at the moment you can get some decent still results from the
visualisation tools:

![DNA in Blender](https://cloud.githubusercontent.com/assets/2887977/22364140/936c16d0-e46f-11e6-9e71-ed8c512663ea.png)

_____

Also, a shout out to the blender DNA example by George Lydecker where
I first saw Blender being used to render DNA, and whose example code
inspired some of what is here. (https://github.com/glydeck/MoloculeParser)
