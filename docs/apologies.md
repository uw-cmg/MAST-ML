
# `ipynb_maker`

This is a wrapper that you put above a plot that attempts to create python
source, bundled into a jupyter notebook, saved to the same place as the plot.
When ran, this notebook will call up the original plotting function (included
in full) and save the image to a `.png` file and display the generated image in
the notebook. `ipynb_maker.py` uses `inspect.getsource` to get the source of
the plotting functions. To get the top chunk of `plot_maker.py` we do something
really... 'neat'.

`plot_helper.py` line 34

```python 
# HEADERENDER don't delete this line, it's used by ipynb maker
```

That sure is __strange__, what is this strange comment that pleads to remain in
existence?

`ipynb_maker.py` line 50-56

```python 
# get source of the top of plot_helper.py
header = ""
with open(plot_helper.__file__) as f:
    for line in f.readlines():
        if 'HEADERENDER' in line:
            break
        header += line
```

Oh... huh. Wow.

