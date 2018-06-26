"""
Module for creating Jupyter Notebooks so user can modify and regenerate the plots
"""

import inspect
import os
import textwrap

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
        if 'savepath' not in all_args:
            full_path = os.path.join(all_args['outdir'], plot_func.__name__ + '.png')
        else:
            full_path = all_args['savepath']

        basename = os.path.basename(full_path) # fix absolute path problem


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
            arg_assignments.append(f'{key} = {repr(var)}')
            arg_names.append(key)
        args_block = ("from numpy import array\n" +
                      "from collections import OrderedDict\n" +
                      '\n'.join(arg_assignments))
        arg_names = ', '.join(arg_names)


        main = textwrap.dedent(f"""\
            import pandas as pd
            from IPython.core.display import Image as image


            {plot_func.__name__}({arg_names})
            image(filename='{basename}')
        """)

        nb = nbformat.v4.new_notebook()
        readme_cell = nbformat.v4.new_markdown_cell(readme)
        text_cells = [header, func_strings, plot_func_string, args_block, main]
        cells = [readme_cell] + [nbformat.v4.new_code_cell(cell_text) for cell_text in text_cells]
        nb['cells'] = cells
        nbformat.write(nb, full_path + '.ipynb')

        return plot_func(*args, **kwargs)
    return wrapper
