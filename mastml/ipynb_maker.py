import textwrap
from . import plot_helper
#from mastml.plot_helper import * #parse_stat, plot_stats, make_fig_ax, plot_predicted_vs_true
import nbformat
import inspect
import os

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
        full_path = all_args['savepath']
        filename = os.path.basename(all_args['savepath']) # fix absolute path problem
        all_args['savepath'] = filename

        assert 'savepath' in all_args

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
            image(filename='{filename}')
        """)

        nb = nbformat.v4.new_notebook()
        text_cells = [header, func_strings, plot_func_string, args_block, main]
        cells = [nbformat.v4.new_code_cell(cell_text) for cell_text in text_cells]
        nb['cells'] = cells
        nbformat.write(nb, full_path + '.ipynb')

        return plot_func(*args, **kwargs)
    return wrapper
