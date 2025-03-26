# MACE-MP-0
Investigating sodium ion diffusion in a graphene/MXene heterostructure using MACE-MP-0, MACE-MPA-0 and MACE-OMAT-0 foundation machine learning potentials (MLPs).

### Analysis notebooks
- `mxene_setup.ipynb`: MXene, graphene and heterostructure setup + geometry optimisation
- `phonons.ipynb`: phonons with mp-0, mpa-0 and omat-0
- `adhesion_energy.ipynb`: adhesion energy
- `interlayer_spacing.ipynb`, `interlayer_spacing.py`: interlayer spacing
- `graphene_mxene_basic_properties.ipynb`: visualisation of heterostructures + bond length and thickness calculations
- `na_diffusion_setup.ipynb`, `na_structure_figures.ipynb`: Na diffusion setup, visualisation of Na intercalation structures
- `na_analysis.ipynb`: Na analysis: interlayer spacing heatmap, diffusion path, MSD, open circuit voltage

### CSD3 scripts
- simulating heterostructues, heterostrucutres with 1 Na ion and heterostrucutres with 8 Na ions
    - `mace_omat_MG.py`
    - `mace_omat_MG_na.py`
    - `mace_omat_MG_8na.py`
- example slurm script used to run bash and python scripts
    - `my_slurm_submit.wilkes3`
