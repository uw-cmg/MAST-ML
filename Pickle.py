import pickle
import numpy as np
import pandas as pd
from mastml.mastml import Mastml
from mastml.feature_generators import ElementalFeatureGenerator

pfile_X = "Codes/bandgap_pbe_X.pickle"
pfile_Y = "Codes/bandgap_pbe_Y.pickle"

with open(pfile_X, 'rb') as f:
    x = pickle.load(f)

# for k,v in x.items():
    # print(k)

# n = 74992
# acc = 0
# for id in x['icsd_id']:
#     if not id:
#         acc += 1

# print(acc / n)
# print(list(numpy.unique(x['reference'])))
# print(list(numpy.unique(x['comments'])))
# print(list(numpy.unique(x['bandgap type'])))
# print(list(numpy.unique(x['comp method'])))

x_clean = x[['composition', 'structure', 'space group']]

composition = x_clean['composition']
structure = x_clean['structure']

m = 1

foo = structure[m]
bar = composition[m]

sites = foo['sites']
print("bar", bar)

l = []
for s in sites:
    l.append(s['species'])

element_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# print(bar)
# decomposeComposition(bar)
# print(bar)
# EnsembleModelFeatureSelector()
# generate_elementfraction_features(bar)
v = ElementalFeatureGenerator(pd.DataFrame([bar]))
print("v", v)
