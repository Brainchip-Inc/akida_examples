# Online documentation
Please refer to the [online documentation](https://doc.brainchipinc.com/) for
installation steps, user guides and APIs.

# Building the documentation
Documentation is based on [Sphinx](https://www.sphinx-doc.org/en/master/) and
can be generated locally from the repository top.

## Requirements:
* Python version: python 3.6 to 3.8
* Tensorflow version >= 2.2.0
* Sphinx version 2.1 or above

## Build steps
* *Optional*: create a virtual environment using
[Virtualenv](https://virtualenv.pypa.io/en/latest/) or
[Conda](https://docs.conda.io/en/latest/)

* Install akida packages and their requirements using the provided file:
~~~~
pip install -r requirements.txt
~~~~

* Install Sphinx additional modules used by the documentation:
~~~~
pip install sphinx_rtd_theme
pip install sphinx-gallery
pip install autodocsumm
~~~~

* Finally build using *sphinx-build* tool, refer to
[Sphinx tutorial](https://matplotlib.org/sampledoc/) for details on this command
line:
~~~~
sphinx-build -D plot_gallery=0 -b html ./doc ./build
~~~~

Note that by removing the *-D plot_gallery=0* option in the previous command,
examples will be executed for a nicer rendering but it will take much more time.

* Open *.build/index.html* once done

## Style check
~~~~
yapf --style="{based_on_style:google}" -dr examples/
~~~~

Use the '-i' option instead of '-d' to apply formatting to a file.
