********************************************
Linux or Mac Shell Installation Instructions
********************************************

1. Install conda: ``https://www.anaconda.com/download/``
2. Make an empty environment: ``conda create --name testml``
3. Enter the environment (bash required): ``source activate testml``
4. Use a local python environment: ``conda install python nbformat``
5. Run ``python --version`` to make sure it's 3 not 2
6. Download repo: ``git clone https://github.com/uw-cmg/mast-ml mastml``
7. ``cd mastml``
8. Checkout a good branch: ``git checkout ryan_updates_2018-03-08`` or whatever boss says
9. If you're on MacOS, then make file ``~/.matplotlib/matplotlibrc`` with contents ``backend: TkAgg``
10. ``python setup.py build``
11. ``python setup.py install``
12. ``cd examples``
13. ``python ../MASTML.py example_input.conf``
14. Open ``example_results/index.html`` in your browser and see your pretty plot!

