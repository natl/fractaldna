fractaldna
===
Python routines for generating simple models of DNA
---

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
