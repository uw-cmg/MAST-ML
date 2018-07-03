import unittest
import random
import warnings
import logging
import textwrap
import nbformat
import inspect
from io import StringIO
from pprint import pprint
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from mastml import plot_helper, conf_parser
import mastml.utils
from mastml.legos import feature_generators
from mastml.legos.randomizers import Randomizer
from mastml.legos.feature_normalizers import MeanStdevScaler

#mastml.utils.activate_logging()

class TestLogging(unittest.TestCase):
    """ only run one at a time or you might get like double logging or something """
    def test_log_levels(self):
        mastml.utils.activate_logging()

        warnings.warn('a warning messgage!')

        logging.debug("logging, A DEBUG message")
        logging.info("logging, An INFO message")
        logging.warning("logging, A WARNING message")
        logging.error("logging, An ERROR message")
        logging.critical("logging, A CRITICAL message")

        raise Exception('oooops an exception')

    def test_try_catch_error_log(self):
        mastml.utils.activate_logging()

        try:
            warnings.warn('a warning messgage!')

            logging.debug("logging, A DEBUG message")
            logging.info("logging, An INFO message")
            logging.warning("logging, A WARNING message")
            logging.error("logging, An ERROR message")
            logging.critical("logging, A CRITICAL message")

            raise Exception('oooops an exception')
        except Exception as e:
            with open('errors.log', 'a') as f:
                f.write("really baddddd errror: " + str(e))
            raise e

class TestFeatureSelectionDefaults(unittest.TestCase):
    class_conf = '''
        [FeatureSelection]
            [[SelectFromModel_1]]
            [[SelectKBest_1]]
            [[SelectFromModel_2]]
                estimator = DecisionTreeClassifier
            [[SelectKBest_2]]
                score_func = chi2
        [Models]
            [[DecisionTreeClassifier]]
    '''

    regress_conf = '''
        [FeatureSelection]
            [[SelectFromModel_1]]
            [[SelectKBest_1]]
            [[SelectFromModel_2]]
                estimator = LassoCV
            [[SelectKBest_2]]
                score_func = mutual_info_regression
        [Models]
            [[Ridge]]
    '''
    def test_classification(self):
        conf = conf_parser.parse_conf_file(string_to_filename(self.class_conf))
        pprint(conf)

    def test_regression(self):
        conf = conf_parser.parse_conf_file(string_to_filename(self.regress_conf))
        pprint(conf)

class TestPlotToPython(unittest.TestCase):
    """ How to convert a call to plot to a .py file that the user can modify """
    def test_test(self):
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
    # TODO test for bad api key using pymatgen.ext.matproj.MPRestError
    def test_magpie(self):
        df = pd.read_csv('tests/csv/feature_generation.csv')
        magpie = feature_generators.Magpie('MaterialComp')

        magpie.fit(df)
        df = magpie.transform(df)
        df.to_csv('magpie_test.csv')

    def test_materials_project(self):
        df = pd.read_csv('tests/csv/common_materials.csv')
        materials_project = feature_generators.MaterialsProject('Material', 'TtAHFCrZhQa7cwEy')

        materials_project.fit(df)
        df = materials_project.transform(df)
        df.to_csv('materials_project.csv')

    def test_citrine(self):
        df = pd.read_csv('tests/csv/feature_generation.csv')
        citrine = feature_generators.Citrine('MaterialComp', 'amQVQutFrr7etr4ufQQh0gtt')

        citrine.fit(df)
        df = citrine.transform(df)
        df.to_csv('citrine.csv')

    def test_clean_data(self):
        good = pd.DataFrame([
            [10,20,30,40],
            [50,60,70,80],
            [90,10,20,30],
            [40,50,60,70]])
        missing_row = pd.DataFrame([
            ['','','',''],
            [50,60,70,80],
            [90,10,20,30],
            [40,50,60,70]])
        missing_some = pd.DataFrame([
            [10,'',20,30],
            [50,60,'',80],
            [90,10,20,30],
            [40,50,60,70]])
        missing_col = pd.DataFrame([
            ['',10,10,10],
            ['',60,70,80],
            ['',10,20,30],
            ['',50,60,70]])
        for df in [good, missing_row, missing_some, missing_col]:
            print('before:\n', df, sep='')
            print('aftere:\n', feature_generators.clean_dataframe(df), sep='')
            print()

class TestPlots(unittest.TestCase):
    """ don't mind the mismatched naming conventions for [true actual y_true] and [pred prediction
    y_pred] """

    def setUp(self):

        np.random.seed(1)
        # TODO: update this test
        self.stats = dict([
            ('foo', 500000),
            ('bar', 123.45660923904908),
            ('baz', 'impossoble'),
            ('rosco', 123e-500),
            ('math', r"My long label with $\sqrt{2}$ $\Sigma_{C}$ math" "\n"
                r"continues here with $\pi$"),
        ])

        self.stats2 = dict([
            ('Mean over 10 tests', None),
            ('5-fold average RMSE', (0.27, 0.01)),
            ('5 fold mean error', (0.00 , 0.01)),
            ('R-squared', 0.97),
            ('R-squared (no int)', 0.97)
        ])

        self.y_true = np.random.random(20) * 20
        self.y_pred = self.y_true + np.random.normal(0, 1, 20)
        self.y_pred_list = np.array([np.random.normal(x, 8, np.random.randint(0,10))
                                    for x in self.y_true])

        self.xs = np.random.normal(4, 2, 100)
        self.ys = np.random.normal(7, 1, 100)
        self.zs = np.random.normal(0, 3, 100)
        self.heats = np.random.random(100)

    def test_confusion_matrix(self):
        y_true = np.random.randint(4, size=(50,))
        y_pred = y_true.copy()
        slices = [not bool(x) for x in np.random.randint(3, size=50)]
        y_pred[slices] = [random.randint(1, 3) for s in slices if s]

        names = ['a', 'b', 'c', 'f']
        y_pred = np.array([names[x] for x in y_pred] + ['a', 'a', 'a'])
        y_true = np.array([names[x] for x in y_true] + ['b', 'b', 'b'])

        plot_helper.plot_confusion_matrix(y_true, y_pred, 'results/confuse.png', self.stats)

    def test_predicted_vs_true(self):
        y_pred_tall = 10 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_pred_fat = 100 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_true_tall = 100 * np.arange(90)
        y_true_fat = 10*  np.arange(90)

        plot_helper.plot_predicted_vs_true((y_true_tall, y_pred_tall, self.stats2),
                                           (y_true_fat, y_pred_fat, self.stats2), 'results')

    def test_best_worst(self):
        y_true = np.arange(90)
        y_pred = np.arange(90) + 9*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        y_pred_bad = 0.5*np.arange(90) + 20*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        best_run = dict(y_test_true=y_true, y_test_pred=y_pred, test_metrics=self.stats)
        worst_run = dict(y_test_true=y_true, y_test_pred=y_pred_bad, test_metrics=self.stats2)
        plot_helper.plot_best_worst(best_run, worst_run, 'results/best_worst.png', self.stats2, title='mest morst Overlay')

    def test_residuals_histogram(self):
        plot_helper.plot_residuals_histogram(self.y_true, self.y_pred, 'results/residuals.png', self.stats)

    def test_target_histogram(self):
        y_df = pd.Series(np.concatenate([np.random.normal(4, size=(100,)),
                                         np.random.normal(-20, 10, size=(100,))]))
        plot_helper.plot_target_histogram(y_df, savepath = 'results/target_hist.png')

    def test_predicted_vs_true_bars(self):
        plot_helper.plot_predicted_vs_true_bars(self.y_true, self.y_pred_list, 'results/bars.png', title='test best worst with bars')

    def test_violin(self):
        plot_helper.plot_violin(self.y_true, self.y_pred_list, 'results/violin.png', title='violin.png')

    def test_best_worst_per_point(self):
        plot_helper.plot_best_worst_per_point(self.y_true, self.y_pred_list, 'results/best_worst_per_point.png')

    def test_1d_heatmap(self):
        plot_helper.plot_1d_heatmap(self.xs, self.heats, 'results/1d_heatmap.png')

    def test_2d_heatmap(self):
        plot_helper.plot_2d_heatmap(self.xs, self.ys, self.heats, 'results/2d_heatmap.png')

    def test_3d_heatmap(self):
        plot_helper.plot_3d_heatmap(self.xs, self.ys, self.zs, self.heats,
                                    'results/3d_heatmap.png', r'$\alpha$',
                                    r'$\beta$', r'$\gamma$', 'accuracy')

    def test_scatter(self):
        plot_helper.plot_scatter(self.y_true, self.y_pred, 'results/scatter.png')

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


def string_to_filename(st):
    f = NamedTemporaryFile(mode='w', delete=False)
    f.write(st)
    f.close()
    print('nnn', f.name)
    return f.name
