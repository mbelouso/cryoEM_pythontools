# cryoEM_pythontools
A collection of Jupyter notebooks for representation of cryoEM data and teaching tools

required python version: 3.10

General requirements:
matplotlib
scipy
numpy
mrcfile
ipywidgets
joblib
pyvista
pyvista[jupyter]


# Recommended to create an anacoda enviroment:

conda create --name cryoEM_pythontools python=3.10

conda activate cryoEM_pythontools

pip install matplotlib scipy numpy mrcfile ipywidgets joblib pyvista pyvista[juptyer]

# Notes: On Linux (only system really tested):

You may need to install some openGL stuff for the pyvista 3D plots to work:

sudo apt install xvfb




