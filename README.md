# cryoEM_pythontools

A collection of Jupyter notebooks for analysis, visualisation, and teaching of
cryoEM single-particle data.

---

## Conda environments

Two separate Anaconda environments are used, depending on the notebook.

---

### 1 · `cryoEM_dynamics` — FSC analysis and map tools

Used for:
- `FSC_Analysis.ipynb` — Fourier Shell Correlation between 3-D MRC volumes
- `Map_BackProjection.ipynb`
- `parse_thermo_xml.ipynb`

**Python version:** 3.12

**Create and activate the environment:**

```bash
conda create --name cryoEM_dynamics python=3.12
conda activate cryoEM_dynamics
```

**Install required packages:**

```bash
pip install mrcfile numpy scipy matplotlib seaborn pandas ipywidgets
```

**Register the environment as a Jupyter kernel** (so it appears in VS Code / JupyterLab):

```bash
pip install ipykernel
python -m ipykernel install --user --name cryoEM_dynamics --display-name "cryoEM_dynamics"
```

**Verify all packages are present:**

```bash
pip show mrcfile numpy scipy matplotlib seaborn pandas
```

---

### 2 · `cryoEM_pythontools` — 3-D visualisation and interactive widgets

Used for notebooks that require PyVista 3-D rendering and ipywidgets.

**Python version:** 3.10

**Create and activate the environment:**

```bash
conda create --name cryoEM_pythontools python=3.10
conda activate cryoEM_pythontools
```

**Install required packages:**

```bash
pip install matplotlib scipy numpy mrcfile ipywidgets joblib pyvista "pyvista[jupyter]"
```

**Register the Jupyter kernel:**

```bash
pip install ipykernel
python -m ipykernel install --user --name cryoEM_pythontools --display-name "cryoEM_pythontools"
```

**Notes — Linux only (primary tested platform):**

PyVista 3-D plots require OpenGL. If running on a headless server install:

```bash
sudo apt install xvfb
```

and prefix notebook launches with `xvfb-run`.

---

### 3 · `cryoDRGN` — Latent space analysis

Used for:
- `cryoDRGN_LatentSPACE_Analysis.ipynb`

Follow the official cryoDRGN setup instructions:
<https://github.com/ml-struct-bio/cryodrgn>

Then, inside the activated cryoDRGN environment, additionally install:

```bash
conda install tensorflow-gpu
```

---

## FSC Analysis — quick start

```python
# In FSC_Analysis.ipynb, set the maps directory and run all cells:
MAPS_DIR = "maps"        # folder containing *.mrc volumes
XLIM     = (8.0, 2.0)   # resolution axis range in Å
```

All unique pairwise FSC curves are computed automatically. Results are saved to
`fsc_output/` as per-pair CSVs and a `fsc_summary.csv`.


