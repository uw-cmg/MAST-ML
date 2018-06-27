"""
Module for creating Jupyter Notebooks so user can modify and regenerate the plots
This whole thing is a hack. But it's the only way, short of repeating every line in plot_helper
twice.
"""

import inspect
import os
import textwrap
from pandas import DataFrame, Series

import nbformat

from . import plot_helper # TODO: fix cyclic import

def ipynb_maker(plot_func):
    """
    wraps a plotting func so it also outputs it's own usable source
    """

    def wrapper(*args, **kwargs):

        # convert everything to kwargs for easier display
        # from geniuses at https://stackoverflow.com/a/831164
        #kwargs.update(dict(zip(plot_func.func_code.co_varnames, args)))
        sig = inspect.signature(plot_func)
        binding = sig.bind(*args, **kwargs)
        all_args = binding.arguments

        # if this is an outdir style function, fill in savepath and delete outdir
        if 'savepath' in all_args:
            ipynb_savepath = all_args['savepath']
            knows_savepath = True
            basename = os.path.basename(ipynb_savepath) # fix absolute path problem
        elif 'outdir' in all_args:
            knows_savepath = False
            basename = plot_func.__name__
            ipynb_savepath = os.path.join(all_args['outdir'], basename)
        else:
            raise Exception('you must have an "outdir" or "savepath" argument to use ipynb_maker')



        readme = textwrap.dedent(f"""\
            This notebook was automatically generated from your MAST-ML run so you can recreate the
            plots. Some things are a bit different from the usual way of creating plots - we are
            using the [object oriented
            interface](https://matplotlib.org/tutorials/introductory/lifecycle.html) instead of
            pyplot to create the `fig` and `ax` instances. 
        """)

        # get source of the top of plot_helper.py
        header = ""
        with open(plot_helper.__file__) as f:
            for line in f.readlines():
                if 'HEADERENDER' in line:
                    break
                header += line

        core_funcs = [plot_helper.parse_stat, plot_helper.plot_stats, plot_helper.make_fig_ax]
        func_strings = '\n\n'.join(inspect.getsource(func) for func in core_funcs)

        plot_func_string = inspect.getsource(plot_func)
        # remove first line that has this decorator on it (!!!)
        plot_func_string = '\n'.join(plot_func_string.split('\n')[1:])

        # put the arguments and their values in the code
        arg_assignments = []
        arg_names = []
        for key, var in all_args.items():
            if isinstance(var, DataFrame):
                # this is amazing
                arg_assignments.append(f"{key} = pd.read_csv(StringIO('''\n{var.to_csv(index=False)}'''))")
            elif isinstance(var, Series):
                arg_assignments.append(f"{key} = pd.Series(pd.read_csv(StringIO('''\n{var.to_csv(index=False)}''')).iloc[:,0])")
            else:
                arg_assignments.append(f'{key} = {repr(var)}')
            arg_names.append(key)
        args_block = ("from numpy import array\n" +
                      "from collections import OrderedDict\n" +
                      "from io import StringIO\n" +
                      '\n'.join(arg_assignments))
        arg_names = ', '.join(arg_names)


        if knows_savepath:
            main = textwrap.dedent(f"""\
                import pandas as pd
                from IPython.display import Image, display


                {plot_func.__name__}({arg_names})
                display(Image(filename='{basename}'))
            """)
        else:
            main = textwrap.dedent(f"""\
                import pandas as pd
                from IPython.display import Image, display

                plot_paths = predicted_vs_true(train_triple, test_triple, outdir)
                for plot_path in plot_paths:
                    display(Image(filename=plot_path))
            """)

        nb = nbformat.v4.new_notebook()
        readme_cell = nbformat.v4.new_markdown_cell(readme)
        text_cells = [header, func_strings, plot_func_string, args_block, main]
        cells = [readme_cell] + [nbformat.v4.new_code_cell(cell_text) for cell_text in text_cells]
        nb['cells'] = cells
        nbformat.write(nb, ipynb_savepath + '.ipynb')

        return plot_func(*args, **kwargs)
    return wrapper


