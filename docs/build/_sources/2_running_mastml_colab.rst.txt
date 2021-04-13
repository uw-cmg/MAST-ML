******************************************************************
Running MAST-ML on Google Colab
******************************************************************

In addition to running MAST-ML on your own machine or computing cluster, MAST-ML
can be run using cloud resources on Google Colab. This can be advantageous as you
don't have to worry about installing MAST-ML yourself, and all output files can be
saved directly to your Google Drive.

MAST-ML comes with a notebook called MASTML_Colab.ipynb that you can open in Google Colab

:download:`MASTML_Colab.ipynb <MASTML_Colab.ipynb>`

Once you open the notebook in Google Colab, it will look something like this:

.. image:: ColabHome.png

There are a few blocks of code in this notebook. The first block performs a pip install of MAST-ML for this
Colab session. The second block links your Google Drive to the Colab instance so MAST-ML can save your run
output directly to your Google Drive.

The one thing you'll need to do from here is to upload a data file (.csv or .xlsx format) and MAST-ML
input file (.conf format) to this Colab session. Files can be uploaded by pressing the vertical arrow
on the left side of the screen, by the file directory tree.

:download:`example_input.conf <example_input.conf>`

:download:`example_data.xlsx <example_data.xlsx>`

Note that when a Colab session ends, the files you upload will be deleted. Since your output will be saved
to your Google Drive, the data an input files will be deleted. Note that MAST-ML automatically saves a copy
of both of these files to your output directory for each run you do.
