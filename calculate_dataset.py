__author__ = 'haotian'

import data_parser as dp


data = dp.parse('../DBTT/DBTT_Data14.5.csv')
data.normalization(
    ['Cu (At%)', 'Ni (At%)', 'Mn (At%)', 'P (At%)','Si (At%)','C (At%)'],
    normalization_type='t')
data.normalization(['log(fluence)','log(eff fluence)','log(flux)','Temp (C)','log(time)'])
data.std_normalization(['delta sigma', 'EONY predicted', 'CD predicted (Mpa)'])
data.output('../DBTT/DBTT_Data15.csv')
