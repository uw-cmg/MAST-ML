import unittest
import random

import pandas as pd
import numpy as np

from mastml import mastml
from mastml import plot_helper, html_helper
from mastml.legos.randomizers import Randomizer
from mastml.legos.feature_normalizers import MeanStdevScaler
from matplotlib.ticker import MaxNLocator
from mastml.legos import feature_generators

class SmokeTests(unittest.TestCase):

    def test_classification(self):
        mastml.mastml_run('tests/conf/classification.conf', 'tests/csv/three_clusters.csv', 'results/classification')

    def test_regression(self):
        mastml.mastml_run('tests/conf/regression.conf', 'tests/csv/boston_housing.csv', 'results/regression')

    def test_generation(self):
        mastml.mastml_run('tests/conf/feature-gen.conf', 'tests/csv/feature-gen.csv', 'results/generation')

    # TODO: add the other test conf files and csv files

class TestPlotToPython(unittest.TestCase):
    """ How to convert a call to plot to a .py file that the user can modify """
    def test_test(self):
        import textwrap
        header = textwrap.dedent("""\
            import os.path
            import itertools

            import numpy as np
            import matplotlib
            from matplotlib import pyplot as plt
            from sklearn.metrics import confusion_matrix

            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure, figaspect
            from matplotlib.ticker import MaxNLocator

            # set all font to bigger
            font = {'size'   : 18}
            matplotlib.rc('font', **font)

            # turn on autolayout (why is it not default?)
            matplotlib.rc('figure', autolayout=True)
        """)

        from mastml.plot_helper import parse_stat, plot_stats, make_fig_ax, plot_predicted_vs_true
        import nbformat
        import inspect

        core_funcs = [parse_stat, plot_stats, make_fig_ax]
        func_strings = '\n\n'.join(inspect.getsource(func) for func in core_funcs)

        plot_func = plot_predicted_vs_true
        plot_func_string = inspect.getsource(plot_func)

        csv_file = 'tests/csv/predicted_vs_measured.csv'
        stats = [('some stats',),
                 ('foo', 20)]



        main = textwrap.dedent(f"""\
            import pandas as pd
            from IPython.core.display import Image as image

            df = pd.read_csv('{csv_file}')
            y_true = df['Enorm DFT (eV)'].values
            y_pred = df['Enorm Predicted (eV)'].values
            savepath = './foobar.png'
            stats = {stats}

            {plot_func.__name__}(y_true, y_pred, savepath, stats, title='some plot of some data')
            image(filename='foobar.png')
        """)

        nb = nbformat.v4.new_notebook()
        text_cells = [header, func_strings, plot_func_string, main]
        cells = [nbformat.v4.new_code_cell(cell_text) for cell_text in text_cells]
        nb['cells'] = cells
        nbformat.write(nb, 'test.ipynb')

class TestGeneration(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('tests/csv/feature_generation.csv')

    def test_magpie(self):
        magpie = feature_generators.Magpie('MaterialComp')
        df = self.df.copy()
        magpie.fit(df)
        df = magpie.transform(df)
        df.to_csv('magpie_test.csv')

    def test_materials_project(self):
        materials_project = feature_generators.MaterialsProject('MaterialComp', 'TtAHFCrZhQa7cwEy')
        #materials_project = feature_generators.MaterialsProject('MaterialComp',  'amQVQutFrr7etr4ufQQh0gtt')
        # TODO test for bad api key using pymatgen.ext.matproj.MPRestError

        df = self.df.copy()
        materials_project.fit(df)
        df = materials_project.transform(df)
        df.to_csv('materials_project.csv')

    def test_citrine(self):
        citrine = feature_generators.Citrine('MaterialComp', 'amQVQutFrr7etr4ufQQh0gtt')
        df = self.df.copy()
        citrine.fit(df)
        df = citrine.transform(df)
        df.to_csv('citrine.csv')


class TestPlots(unittest.TestCase):
    """ don't mind the mismatched naming conventions for [true actual y_true] and [pred prediction
    y_pred] """

    def setUp(self):
        # TODO: update this test
        self.stats = [
            ('foo', 500000),
            ('bar', 123.45660923904908),
            ('baz', 'impossoble'),
            ('rosco', 123e-500),
            ('math', r"My long label with $\sqrt{2}$ $\Sigma_{C}$ math" "\n"
                r"continues here with $\pi$"),
        ]

        self.stats2 = [
            ('Mean over 10 tests',),
            ('5-fold average RMSE', 0.27, 0.01),
            ('5 fold mean error', 0.00 , 0.01),
            ('R-squared', 0.97),
            ('R-squared (no int)', 0.97)
        ]

    def test_plot_predicted_vs_true(self):
        y_pred_tall = 10 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_pred_fat = 0.1 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_true = np.arange(90)

        plot_helper.plot_predicted_vs_true(y_true, y_pred_tall, 'pred-vs-true_skinny.png', self.stats2)
        plot_helper.plot_predicted_vs_true(y_true, y_pred_fat,  'pred-vs-true_fat.png',    self.stats2)

    def test_residuals_histogram(self):
        y_pred = np.arange(30) + sum(np.random.random_sample((30,)) for _ in range(10)) - 3
        y_true = np.arange(30)
        plot_helper.plot_residuals_histogram(y_true, y_pred, 'residuals.png', self.stats)

    def test_confusion_matrix(self):
        y_true = np.random.randint(4, size=(50,))
        y_pred = y_true.copy()
        slices = [not bool(x) for x in np.random.randint(3, size=50)]
        y_pred[slices] = [random.randint(1, 3) for s in slices if s]

        names = ['a', 'b', 'c', 'f']
        y_pred = np.array([names[x] for x in y_pred] + ['a', 'a', 'a'])
        y_true = np.array([names[x] for x in y_true] + ['b', 'b', 'b'])

        plot_helper.plot_confusion_matrix(y_true, y_pred, 'confuse.png', self.stats)

    def test_best_worst(self):
        y_true = np.arange(90)
        y_pred = np.arange(90) + 9*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        y_pred_bad = 0.5*np.arange(90) + 20*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        plot_helper.plot_best_worst(y_true, y_pred, y_true, y_pred_bad, 'best-worst.png', self.stats2)



class TestHtml(unittest.TestCase):

    def test_image_list(self):
        #imgs = ['cf.png', 'rh.png', 'pred-vs-true.png']
        #html_helper.make_html(imgs, 'tests/csv/three_clusters.csv', 'tests/conf/fullrun.conf', 'oop.txt', './')
        html_helper.make_html('results/regression_test')
        #html_helper.make_html('results/generation')
        #html_helper.make_html('results/classification')
        #html_helper.make_html('results/regression')


class TestRandomizer(unittest.TestCase):

    def test_shuffle_data(self):
        d = pd.DataFrame(np.arange(10).reshape(5,2))
        d.columns = ['a', 'b']
        r = Randomizer('b')
        r.fit(d)
        buffler = r.transform(d)
        good = r.inverse_transform(buffler)
        # this has a 1/120 chance of failing unexpectedly.
        self.assertFalse(d.equals(buffler))
        self.assertTrue(d.equals(good))

class TestNormalization(unittest.TestCase):

    def test_normalization(self):
        d1 = pd.DataFrame(np.random.random((7,3)), columns=['a','b','c']) * 4 + 6 # some random data

        fn = MeanStdevScaler(features=['a','c'], mean=-2, stdev=5)
        fn.fit()

        d2 = fn.transform(d1)
        self.assertTrue(set(d1.columns) == set(d2.columns))
        arr = d2[['a','c']].values
        self.assertTrue(abs(arr.mean() - (-2)) < 0.001)
        self.assertTrue(abs(arr.std() - 5) < 0.001)

        d3 = fn.inverse_transform(d2)
        self.assertTrue(set(d1.columns) == set(d3.columns))
        self.assertTrue((abs(d3 - d1) < .001).all().all())
