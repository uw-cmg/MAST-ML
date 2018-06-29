import unittest
import random
import warnings
import logging
from tempfile import NamedTemporaryFile
from pprint import pprint

import pandas as pd
import numpy as np

from mastml import mastml
from mastml import plot_helper, html_helper
from mastml.legos.randomizers import Randomizer
from mastml.legos.feature_normalizers import MeanStdevScaler
from mastml.legos import feature_generators

#mastml.utils.activate_logging()

class SmokeTests(unittest.TestCase):

    def test_classification(self):
        mastml.mastml_run('tests/conf/classification.conf', 'tests/csv/three_clusters.csv', 'results/classification')

    def test_regression(self):
        mastml.mastml_run('tests/conf/regression.conf', 'tests/csv/boston_housing.csv', 'results/regression')

    def test_generation(self):
        mastml.mastml_run('tests/conf/feature-gen.conf', 'tests/csv/feature-gen.csv', 'results/generation')

    # TODO: add the other test conf files and csv files

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
        from io import StringIO
        from mastml import conf_parser

        conf = conf_parser.parse_conf_file(string_to_filename(self.class_conf))
        pprint(conf)

    def test_regression(self):
        from io import StringIO
        from mastml import conf_parser

        conf = conf_parser.parse_conf_file(string_to_filename(self.regress_conf))
        pprint(conf)

def string_to_filename(st):
    f = NamedTemporaryFile(mode='w', delete=False)
    f.write(st)
    f.close()
    print('nnn', f.name)
    return f.name

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

    def test_plot_predicted_vs_true(self):
        y_pred_tall = 10 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_pred_fat = 100 * (np.arange(90) + np.random.random_sample((90,)) * 10 - 3) + 0.5
        y_true_tall = 100 * np.arange(90)
        y_true_fat = 10*  np.arange(90)

        plot_helper.predicted_vs_true((y_true_tall, y_pred_tall, self.stats2), 
                                           (y_true_fat, y_pred_fat, self.stats2), './')
                                           

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

    def test_target_histogram(self):
        y_df = pd.Series(np.concatenate([np.random.normal(4, size=(100,)), 
                                         np.random.normal(-20, 10, size=(100,))]))
        plot_helper.target_histogram(y_df, savepath = 'target_hist.png')


    def test_predicted_vs_true_bars(self):
        y_true = np.arange(90) + 9*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        y_pred_list = [[x + 30*np.random.normal(2, 1) for _ in "r"*np.random.randint(1,5)] for x in y_true]
        plot_helper.predicted_vs_true_bars(y_true, y_pred_list, 'bars.png', title='test best worst with bars')

        y_true = [1,3,5,5,6,7]
        y_pred_list = [[x + np.random.normal(2, 1) for _ in "a"*20] for x in y_true]
        plot_helper.predicted_vs_true_bars(y_true, y_pred_list, 'bars2.png', title='test best worst with bars')

    def test_violin(self):
        y_true = np.arange(90) + 9*sum(np.random.random_sample((90,)) for _ in range(10)) - 54
        y_pred_list = [[x + 30*np.random.normal(2, 1) for _ in "r"*np.random.randint(1,5)] for x in y_true]
        plot_helper.violin(y_true, y_pred_list, 'violin.png', title='violin.png')

        y_true = [1,3,5,5,6,7]
        y_pred_list = [[x + np.random.normal(2, 1) for _ in "a"*20] for x in y_true]
        plot_helper.violin(y_true, y_pred_list, 'violin2.png', title='violin2.png')

    def test_3d_heatmap(self):
        xs = np.random.normal(4, 2, 100)
        ys = np.random.normal(7, 1, 100)
        zs = np.random.normal(0, 3, 100)
        heats = np.random.random(100)

        plot_helper.plot_3d_heatmap(xs, ys, zs, heats, '3d_heatmap.png', r'$\alpha$', r'$\beta$', r'$\gamma$', 'accuracy')


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
