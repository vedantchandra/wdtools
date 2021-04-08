Installation
==================

Downloading and Installing wdtools
++++++++++++++++++++++++++++++++++++

Please note that wdtools is only supported and tested for Python 3.7 and 3.8. The installation will load specific versions of libararies like ``numpy`` and ``tensorflow`` to ensure that all the libraries play well together. Therefore, it is strongly recommended that you install and use ``wdtools`` in its own virtual environment using a manager like ``conda``. If you use ``conda``, make sure to proceed with installation only once you've activated the ``conda`` environment you want to use ``wdtools`` within.

The simplest way to install wdtools is to clone the GitHub repository into any directory of your choice:

.. code-block:: bash

   cd ~/YourDirectoryPath/
   git clone https://github.com/vedantchandra/wdtools.git

You can replace the first line with any directory of your choice.

Next, navigate to this directory and run ``python setup.py install``. This should install the required dependencies and add ``wdtools`` to your Python path.  All the required dependencies are also listed in the ``requirements.txt`` file in the repository.

Once installed, you can add the following line to your Python projects to import wdtools into your workspace:

.. code-block:: python

   import wdtools

If you have any trouble with installation don't hesitate to `raise a new issue <https://github.com/vedantchandra/wdtools/issues>`_.
