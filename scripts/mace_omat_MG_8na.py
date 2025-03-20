from mace.calculators import mace_mp
from ase import build

from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase import Atoms
from ase.build import bulk
from ase.visualize import view
import numpy as np
from ase.build import add_vacuum
from ase.optimize import LBFGS, BFGS
from ase import Atom

from ase.io.trajectory import Trajectory
from ase.md import Langevin, Bussi
from ase import io


macemp_omat = mace_mp(model="/home/jh2536/rds/hpc-work/mace_data/models/mace-omat-0-medium.model", dispersion=True, default_dtype="float64", device='cuda', enable_cueq=True)

#mxene_relaxed = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/relaxed_mxene.xyz")
#graphene_atoms = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/graphene_atoms.xyz")
#go_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/go_omat_relax.xyz")
#goh_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/goh_omat_relax.xyz")

m_g_relaxed_omat = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/m_g_relaxed_flip_omat.xyz")



def add_sodium_many(heterostructure, n_Na=8):
    """
    Adds n sodium ions to the heterostructure in a 3-2-3 pattern
    """

    heterostructure = heterostructure.copy()
    
    # z-coordinates of graphene (C atoms) and MXene (O atoms)
    graphene_C_atoms = [atom.position[2] for atom in heterostructure if atom.tag == 10]
    mxene_O_atoms = [atom.position[2] for atom in heterostructure if atom.tag == 3]
    
    z_mxene = np.mean(mxene_O_atoms)
    z_graphene = np.mean(graphene_C_atoms)

    # Na halfway between MXene and graphene
    na_z = z_mxene + 0.5 * (z_graphene - z_mxene)


    positions = heterostructure.get_positions()
    xmin, ymin = np.min(positions[:, :2], axis=0)
    xmax, ymax = np.max(positions[:, :2], axis=0)


    y_positions = np.linspace(ymin+0.5, ymax-0.5, 3)

    # top and bottom row
    x_positions_top = np.linspace(xmin, xmax, 3, endpoint=False)  # 3 in top row
    x_positions_bottom = np.linspace(xmin+5, xmax+5, 3, endpoint=False)  # 3 in bottom row
    
    # middle row
    x_spacing = (x_positions_top[1] - x_positions_top[0]) / 2
    x_positions_middle = x_positions_top[:-1] + x_spacing 


    na_positions = []
    
    for x in x_positions_top:
        na_positions.append([x, y_positions[2], na_z])

    for x in x_positions_middle:
        na_positions.append([x, y_positions[1], na_z])

    for x in x_positions_bottom:
        na_positions.append([x, y_positions[0], na_z])


    na_atoms = Atoms("Na" * n_Na, positions=na_positions)
    heterostructure += na_atoms 

    return heterostructure


m_g_na_many = add_sodium_many(m_g_relaxed_omat)


#--------------------------------MD--------------------------------

import time
start_time = time.time()

def printenergy(a=m_g_na_many):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    elapsed_time = time.time() - start_time
    elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV Time Elapsed: %dm %.1fs' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, int(elapsed_min), int(elapsed_sec)))


from ase.constraints import FixSubsetCom

    
m_g_na_many.calc = macemp_omat
T_init = 300
MaxwellBoltzmannDistribution(m_g_na_many, temperature_K=T_init)


dyn = Bussi(m_g_na_many, 1 * units.fs, T_init, 0.01)
n_steps = 100000

dyn.attach(printenergy, interval = 1000) 

tags = m_g_na_many.get_tags()

mxene_indices = np.where(np.isin(tags, [0, 1, 2, 3]))[0]  # MXene atoms
graphene_indices = np.where(np.isin(tags, [10, 11, 12]))[0]  # Graphene atoms

fix_com_mxene = FixSubsetCom(indices=mxene_indices)
fix_com_graphene = FixSubsetCom(indices=graphene_indices)
m_g_na_many.set_constraint([fix_com_mxene, fix_com_graphene])


traj = Trajectory('/home/jh2536/rds/hpc-work/mace_data/traj/mg_na_many_omat_100000.traj', 'w', m_g_na_many)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)
