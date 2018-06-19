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

def make_html(save_dir, images: list, starting_data_csv, computed_csvs: list, conf, statistics,
        error_log, debug_log, best=None, median=None, worst=None):
    """ Makes saves html to file with all of the stuff you give it. 
    all arguments refer to the file paths to the things, not the things themselves. """

    # TODO everyhting should be flat files in same dir as index.html
    images = [os.path.basename(path) for path in images]
    statistics = os.path.basename(statistics)

    # check if error_log has an substantial content
    if os.path.exists(error_log):
        with open(error_log) as f:
            errors_present = len(f.read()) > 4

    with document(title='MASTML') as doc:

        # title and date
        h1('MAterial Science Tools - Machine Learning')
        h4(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        # link to error log
        if errors_present:
            p('You have errors! check ', link(error_log))

        link(statistics)

        # best worst and median images
        for name, path in (('best', best), ('median', median), ('worst', worst)):
            if path:
                h2(name)
                h3(path)
                div(img(src=best), _class='photo')

        # all plots
        h2('Plots')

        for path in images:
            h3(path)
            div(img(src=path), _class='photo')

        # links to csv's
        for path in computed_csvs:
            link(path)

        # link to conf file and starting data
        link(conf)
        link(starting_data_csv)
        link(debug_log)


    with open(os.path.join(save_dir, 'index.html'), 'w') as f:
        f.write(doc.render())


def link(href):
    """ Makes it slightly shorter to link files with their names"""
    return p(a(href, href=href))
