import numpy as np


# use tags to identify atoms

# Tags
# 0: mxene Ti
# 1: bottom mxene O
# 2: mxene carbon
# 3: top mxene O
# 10: graphene C
# 11: graphene O
# 12: graphene H


def calculate_interlayer_spacing(atoms, avg_z=False, bottom=False):
    # define interlayer spacing as the distance between avg C z positions in graphene and 
    # the avg O z postions in mxene's O layer closest to graphene
    
    graphene_C_atoms = [atom.position[2] for atom in atoms if atom.tag == 10]

    if bottom == True:
        mxene_O_atoms = [atom.position[2] for atom in atoms if atom.tag == 1]
    else:
        mxene_O_atoms = [atom.position[2] for atom in atoms if atom.tag == 3]


    # average z-positions
    avg_z_C = np.mean(graphene_C_atoms)
    avg_z_O = np.mean(mxene_O_atoms)
    
    if avg_z == True:
        return avg_z_C - avg_z_O, avg_z_C, avg_z_O

    # interlayer spacing (MXene O above Graphene C)
    return avg_z_C - avg_z_O


def set_interlayer_spacing(heterostructure, height):
    heterostructure = heterostructure.copy()
    
    current_spacing, _, avg_z_O = calculate_interlayer_spacing(heterostructure, avg_z=True)
    
    difference = height - current_spacing
    
    # shift every atom above mxene's top O layer (to include functional groups of graphene if present)
    graphene_C_O_H_atoms = [atom for atom in heterostructure if atom.tag in [10, 11, 12]]
    
    for atom in graphene_C_O_H_atoms:
        atom.position[2] += difference
    
    return heterostructure