from setuptools import setup, find_packages

setup(
    name='aptecoPythonUtilities',
    version='0.0',
    description='Python Utilities',
    author='Apteco',
    py_modules=['utils_explore'],
    scripts=['aptecoPythonUtilities/utils_explore.py'],
    python_requires='>=3.5.0',
    install_requires=['pandas'],
)

# To install:
# - Open command prompt as administrator (right click on Start)
# - Go to C:\Users\rkirk\Documents\GIT\Python\TestPyCharm
# - Execute "py setup.py install"

# Has installed .egg file to C:\Users\rkirk\AppData\Local\Programs\Python\Python36-32\Lib\site-packages
# you can install it by pip install -e foo_package. The option -e or --editable installs a project in editable mode
# If you are the author of the package, you can use the flag zip_safe=False in setup.py

# NB: Can use setup to export whole folder (aka package?)

# Can change where command line looks for Python by adding to PATH environment variable
# Add location of Python to the search

