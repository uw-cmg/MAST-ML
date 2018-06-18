import os
from time import gmtime, strftime

import glob
from dominate import document
from dominate.tags import *

# Goal for this thing:
"""
MAST-ML massively machine ml Turing complete
Datetime

if errors___presetn_:
    errorwarings.txt

Best:
Model params
Title
[plot]

Worst:
model params
Title
[plot]

Title
[plot]

Title
[plot]

final_data.csv
statiscics.csv
info.txt
debug.txt
"""

def make_html(image_paths, data_csv_path, conf_path, statistics_path, save_dir):

    with document(title='MASTML') as doc:
        h1('MAterial Science Tools - Machine Learning')
        h4(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        h2('Plots')
        for path in image_paths:
            div(img(src=path), _class='photo')

        p(a('Computed CSV', href=data_csv_path))
        p(a('Conf file', href=conf_path))
        p(a('statistics CSV', href=statistics_path))


    with open(os.path.join(save_dir, 'index.html'), 'w') as f:
        f.write(doc.render())


