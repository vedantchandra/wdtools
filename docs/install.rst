Installation
==================

Downloading and Installing wdtools
++++++++++++++++++++++++++++++++++++

Please note that wdtools is only supported and tested for Python 3.7 and 3.8. The installation will load particular versions of certain libararies like ``numpy`` and ``tensorflow``. Therefore, it is strongly recommended that you use ``wdtools`` in its own virtual environment using a manager like ``conda``. 

The simplest way to install wdtools is to clone the GitHub repository into any directory of your choice:

.. code-block:: bash

   cd ~/YourDirectoryPath/
   git clone https://github.com/vedantchandra/wdtools.git

You can replace the first line with any directory of your choice.

Next, navigate to this directory and run ``python setup.py install``. This should install the required dependencies and add ``wdtools`` to your Python path. If you use ``conda``, make sure to execute this command within the ``conda`` environment you want to use ``wdtools`` within. All the required dependencies are also listed in the ``requirements.txt`` file in the repository.

Once installed, you can add the following line to your Python projects to import wdtools into your workspace:

.. code-block:: python

   import wdtools

If you have any trouble with installation don't hesitate to `raise a new issue <https://github.com/vedantchandra/wdtools/issues>`_.
