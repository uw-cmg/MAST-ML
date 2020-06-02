"""
Module for generating an HTML file, called index.html, which contains an overview of the key data and plots from a
MAST-ML run. Images of cross-validation parity plots, data histograms, data statistics, and links to the relevant files
are all provided.
"""

import os
from os.path import join, relpath # because it's used so much
from time import gmtime, strftime
import logging
from dominate import document
from dominate.tags import *

log = logging.getLogger('mastml')

def make_html(outdir):
    """
    Method used to create the main index.html file

    Args:

        outdir: (str), user-specified output path which designates where all results of MAST-ML run are written

    Returns:

        None

    """

    with document(title='MASTML') as doc:
        # title and date
        h1('MAterial Science Tools - Machine Learning')
        h4(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        # link to error log
        #if errors_present:
        #    p('You have errors! check ', make_link(error_log))

        combos = list()
        link_sections = list()
        #favorites = dict()

        for root, dirs, files in os.walk(outdir):
            # find a folder that contains split_ folder.
            # For example, results/StandardScaler/SelectKBest/LinearRegression/KFold
            for d in dirs:
                if d.startswith('split_0'):
                    combos.append(root)

            # extract links to important csvs and conf
            for f in files:
                csv_whitelist = [
                    'clusters.csv', 'generated_features.csv',
                    'generated_features_no_constant_columns.csv', 'grouped.csv',
                    'input_data_statistics.csv', 'normalized.csv', 'selected.csv', ]
                ext = os.path.splitext(f)[1]
                if f in csv_whitelist or ext in ['.conf', '.log']:
                    link_sections.append(join(root, f))
                    #simple_section(join(root, f), outdir)

        h1('Files')
        for path in link_sections:
            simple_section(path, outdir)

        h1('Plots')

        # show all the images
        for f in os.listdir(outdir):
            if f.endswith('.png'):
                make_image(f, f)

        for combo in combos:
            # come up with a good section title
            path = os.path.normpath(relpath(combo, outdir))
            paths = path.split(os.sep)
            title = " - ".join(paths)
            h2(title)

            # find the best worst overlay
            for fname in os.listdir(combo):
                if fname.endswith('.png'):
                    h3(os.path.splitext(fname)[0]) # probably best_worst overlay
                    make_image(relpath(join(combo, fname), outdir), fname)
                    br()

            # find the split_0 split_1 etc bs stuff
            for fname in os.listdir(combo):
                if fname.startswith('split_'):
                    show_combo(join(combo, fname), outdir)

    with open(join(outdir, 'index.html'), 'w') as f:
        f.write(doc.render())

    log.info('wrote ' + join(outdir, 'index.html'))

def show_combo(combo_dir, outdir):
    """
    Method used to collect combinations of data analysis (e.g. parity plots of train and test data in a CV split) and
    required file paths and display them in the output index.html file.

    Args:

        combo_dir: (str), path containing the relevant data to combine as output in the index.html file

        outdir: (str), user-specified output path which designates where all results of MAST-ML run are written

    Returns:

        None

    """

    # collect test image, train image, and other file links
    links = list()
    train_images = list()
    test_images = list()
    for f in os.listdir(combo_dir):
        if is_train_image(f):
            train_images.append(join(combo_dir, f))
        elif is_test_image(f):
            test_images.append(join(combo_dir, f))
        else:
            links.append(join(combo_dir, f))

    # have a header for split_0 split_1 etc
    h2(combo_dir.split(os.sep)[-1])

    # loop separately so we can control order
    for train_image, test_image in zip(sorted(train_images), sorted(test_images)):
        make_image(relpath(train_image, outdir), 'train')
        make_image(relpath(test_image, outdir), 'test')
        br();br()

    h3('links')
    for l in links:
        make_link(relpath(l, outdir))
        span('  ')

def simple_section(filepath, outdir):
    """
    Method used to create a section name for a particular analysis combination that will be displayed in the index.html file.

    Args:

        filepath: (str), path containing the relevant data to combine as output in the index.html file

        outdir: (str), user-specified output path which designates where all results of MAST-ML run are written

    Returns:

        None

    """

    " Create a section for a combo "
    path = os.path.normpath(relpath(filepath, outdir))
    paths = path.split(os.sep)
    title = " - ".join(paths)
    a(b(title))
    make_link(relpath(filepath, outdir))
    br()

def make_link(href):
    """
    Method used to generate a link to a particular file created from a MAST-ML run. The link will be displayed next to the
    appropriate data or image in the index.html file

    Args:

        href: (str), filename to generate link for

    Returns:

        (dominate.tags html_tag object), hyperlink to filename

    """

    " Make a link where text is filename of href "
    return a(os.path.basename(href), href=href, style='padding-left: 15px;')

def make_image(src, title=None):
    """
    Method used to generate and show an image of a fixed width. The image will be displayed in the appropriate
    section of the index.html file

    Args:

        src: (str), source path of the image to be displayed

        title: (str), title for the image

    Returns:

        None

    """
    " Show an image in fixed width "
    d = div(style='display:inline-block;', _class='photo')
    if title:
        d += h4(title)
        #d += p(a(title))
    d += img(src=src, height='200')

def is_train_image(path):
    """
    Method used to assess whether an image is for training data

    Args:

        path: (str), source path of the image to be displayed

    Returns:

        (bool), True if path is an image (.png) and is for training data (has 'train' in path)

    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[1] == '.png' and 'train' in basename

def is_test_image(path):
    """
    Method used to assess whether an image is for testing data

    Args:

        path: (str), source path of the image to be displayed

    Returns:

        (bool), True if path is an image (.png) and is for testing data (has 'test' in path)

    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[1] == '.png' and 'test' in basename
