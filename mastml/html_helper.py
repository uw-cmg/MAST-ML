"""
Module for generating the nice collected index.html file,
with best/worst plots, links to statistics, and more.
"""

import os
from os.path import join, relpath # because it's used so much
from time import gmtime, strftime

from dominate import document
from dominate.tags import *

def is_train_image(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[1] == '.png' and 'train_' in basename

def is_test_image(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[1] == '.png' and 'test_' in basename

def show_split(split_dir, outdir):

    # collect test image, train image, and other file links
    links = list()
    train_images = list()
    test_images = list()
    for f in os.listdir(split_dir):
        if is_train_image(f):
            train_images.append(join(split_dir, f))
        elif is_test_image(f):
            test_images.append(join(split_dir, f))
        else:
            links.append(join(split_dir, f))

    # have a header for split_0 split_1 etc
    h2(split_dir.split(os.sep)[-1])


    # loop seperately so we can control order
    for train_image, test_image in zip(sorted(train_images), sorted(test_images)):
        make_image(relpath(train_image, outdir), 'train')
        make_image(relpath(test_image, outdir), 'test')
        br();br()

    h3('links')
    for l in links:
        make_link(relpath(l, outdir))
        span('  ')

def simple_section(filepath, outdir):

    # come up with a good section title
    path = os.path.normpath(relpath(filepath, outdir))
    paths = path.split(os.sep)
    title = " - ".join(paths)

    a(b(title))

    make_link(relpath(filepath, outdir))

    br()


def make_html(outdir):
    with document(title='MASTML') as doc:

        # title and date
        h1('MAterial Science Tools - Machine Learning')
        h4(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        # link to error log
        #if errors_present:
        #    p('You have errors! check ', make_link(error_log))

        splits = list()
        link_sections = list()
        #favorites = dict()

        for root, dirs, files in os.walk(outdir):
            # find a folder that contains split_ folder.
            # For example, results/StandardScaler/SelectKBest/LinearRegression/KFold
            for d in dirs:
                if d.startswith('split_'):
                    splits.append(root)

            # extract links to important csvs and conf
            for f in files:
                if (os.path.splitext(f)[1] == '.csv' and f not in ['train.csv', 'test.csv']) or\
                        os.path.splitext(f)[1] == '.conf':
                    link_sections.append(join(root, f))
                    #simple_section(join(root, f), outdir)


        h1('Files')
        for path in link_sections:
            simple_section(path, outdir)

        h1('Plots')
        for split in splits:
            # come up with a good section title
            path = os.path.normpath(relpath(split, outdir))
            paths = path.split(os.sep)
            title = " - ".join(paths)
            h2(title)

            # find the best worst overlay
            for fname in os.listdir(split):
                if fname.endswith('.png'):
                    h3(os.path.splitext(fname)[0]) # probably best_worst overlay
                    make_image(relpath(join(split, fname), outdir), fname)
                    br()

            # find the split_0 split_1 etc bs stuff
            for fname in os.listdir(split):
                if fname.startswith('split_'):
                    show_split(join(split, fname), outdir)






    with open(join(outdir, 'index.html'), 'w') as f:
        f.write(doc.render())

    print('wrote ', join(outdir, 'index.html'))

def make_link(href):
    " Make a link where text is filename of href "
    return a(os.path.basename(href), href=href, style='padding-left: 15px;')

def make_image(src, title=None):
    " Show an image in fixed width "
    d = div(style='display:inline-block;', _class='photo')
    if title:
        d += h4(title)
        #d += p(a(title))
    d += img(src=src, width='500')

