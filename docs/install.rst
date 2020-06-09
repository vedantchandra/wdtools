Installation
==================

Dependencies
+++++++++++++++

We highly recommend installing this package using the ``conda`` utility to ensure all dependencies are loaded with their correct versions. We provide an ``environment.yaml`` file in the root directory of our GitHub repository. To create a new ``conda`` environment with all dependencies, use the following code:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate wdtools

This will install all dependencies and create a new ``conda`` environment called ``wdtools``. Make sure to activate this environment before using wdtools. If you use Jupyter Notebooks, make sure the ``nb_conda`` package is installed to enable environment selection in Jupyter from the 'Kernel' menu.

In case you wish to load the wdtools dependencies into an existing environment (not recommended), first deactivate the environment and then run

.. code-block:: bash

   conda env update --name [your-environment] -f wdtools.yaml

This may cause version conflicts with existing packages in that environment, so we encourage you to use our first method. In case there are any problems with installation, please don't hesitate to contact the authors or `raise a new issue <https://github.com/vedantchandra/wdtools/issues>`_ on the GitHub page. 

Installing wdtools
++++++++++++++++++++

The simplest way to install wdtools is to clone the GitHub repository into any directory of your choice:

.. code-block:: bash

   cd ~/YourDirectoryPath/
   git clone https://github.com/vedantchandra/wdtools.git

You can replace the first line with any directory of your choice. It may be convenient to keep it in a high level directory to make it accesible to all your projects. Add the following lines to your Python projects to import wdtools into your workspace:

.. code-block:: python

   import sys
   sys.path.append('~/wdtools/')
   
   import wdtools

Note that you want to add the *parent* directory (GitHub repository) to the path, so that Python can import the ``wdtools`` package contained within it. And you're done! Again, if you have any trouble with installation don't hesitate to `raise a new issue <https://github.com/vedantchandra/wdtools/issues>`_.