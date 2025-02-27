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


#macemp = mace_mp(dispersion=True, default_dtype="float64")
macemp_omat = mace_mp(model="/Users/joehart/Desktop/0_Cambridge/0_MPhil_Scientific_Computing/Written_assignments/MACE-MP-0/Notebooks_mace/mace_test/mace-omat-0-medium.model", dispersion=True, default_dtype="float64")
#macemp_omat = mace_mp(model="/home/jh2536/rds/hpc-work/mace_data/models/mace-omat-0-medium.model", dispersion=True, default_dtype="float64", enable_cueq=True, device='cuda')

mxene_relaxed = io.read("relaxed_mxene.xyz")
graphene_atoms = io.read("graphene_atoms.xyz")
go_relax = io.read("go_relax.xyz")
goh_relax = io.read("goh_relax.xyz")
# mxene_relaxed = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/relaxed_mxene.xyz")
# graphene_atoms = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/graphene_atoms.xyz")
# go_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/go_relax.xyz")
# goh_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/goh_relax.xyz")







#--------------------------------Create heterostructures--------------------------------


def create_heterostructure(mxene, graphene_layer, height):
    # Stack MXene and graphene-based layer with a given interlayer spacing (height)
    graphene_layer = graphene_layer.copy()
    
    mxene_avg_top_O_z = np.mean([atom.position[2] for atom in mxene if atom.tag == 3])

    #mxene_avg_top_C_z = np.mean([atom.position[2] for atom in mxene if atom.symbol == 'C'])
    #mxene_avg_top_O_z = np.mean([atom.position[2] for atom in mxene if atom.symbol == 'O' and atom.position[2] > mxene_avg_top_C_z])
    
    graphene_layer.positions[:, 2] += mxene_avg_top_O_z - np.average(graphene_layer.positions[:, 2]) + height
    heterostructure = mxene + graphene_layer
    print("MXene avg top O z: ", mxene_avg_top_O_z)

    return heterostructure


m_g = create_heterostructure(mxene_relaxed, graphene_atoms, 4.5)
m_go = create_heterostructure(mxene_relaxed, go_relax, 4.5)
m_goh = create_heterostructure(mxene_relaxed, goh_relax, 4.5)





#--------------------------------Geometry optimisation--------------------------------

m_g_relaxed = m_g.copy()
m_g_relaxed.calc = macemp_omat
optimiser = BFGS(m_g_relaxed)
optimiser.run(fmax=0.001, steps=5000)

# m_go_relaxed = m_go.copy()
# m_go_relaxed.calc = mace_mp()
# optimiser = BFGS(m_go_relaxed)
# optimiser.run(fmax=0.001, steps=5000)

# m_goh_relaxed = m_goh.copy()
# m_goh_relaxed.calc = mace_mp()
# optimiser = BFGS(m_goh_relaxed)
# optimiser.run(fmax=0.001, steps=5000)




#--------------------------------MD--------------------------------

def printenergy(a=m_g_relaxed):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
    
# m_g
#m_g_relaxed.calc = macemp
m_g_relaxed.calc = macemp_omat
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(m_g_relaxed, temperature_K=T_init)


#dyn = Langevin(m_g_relaxed, 1 * units.fs, T_init * units.kB, 0.01)
dyn = Bussi(m_g_relaxed, 0.5 * units.fs, T_init, 0.01)
#n_steps = 5000 + 20000 # 5ps pre-equilibration + 20ps production
n_steps = 2000

dyn.attach(printenergy, interval = 100) 

traj = Trajectory('mg_Bussi2000.traj', 'w', m_g_relaxed)
#traj = Trajectory('/home/jh2536/rds/hpc-work/mace_data/traj/mg_omat_25000.traj', 'w', m_g_relaxed)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)