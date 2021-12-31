from setuptools import setup

setup(name='wdtools',
      version='0.4',
      description='computational tools for the spectroscopic analysis of white dwarfs',
      author='Vedant Chandra',
      author_email='vchandra@jhu.edu',
      license='MIT',
      url='https://github.com/vedantchandra/wdtools',
      package_dir = {},
      packages=['wdtools'],
      package_data={'wdtools':['models/*', 'models/neural_gen/*']},
      dependency_links = [],
      install_requires=['emcee', 'corner', 'tensorflow==2.5.2', 'lmfit', 'scikit-learn', 'numpy==1.19.2'],
      include_package_data=True)