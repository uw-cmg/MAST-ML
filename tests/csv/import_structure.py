from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure
import pandas as pd

STRUCTURE_DIR = '/Users/AlexDo/PycharmProjects/VASP_tools/Al2O3_POSCAR.POSCAR.vasp'
d = [('Al2O3', STRUCTURE_DIR),
      ('GaAs', '/Users/AlexDo/PycharmProjects/VASP_tools/GaAs_POSCAR.POSCAR.vasp')] # I think we want the entire structure object in a single cell of the dataframe
df = pd.DataFrame(d, columns=['Material', 'Structure'], index=None)

df.to_csv('example_matminer.csv')
