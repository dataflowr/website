# JupyterLab

This post explains how to install and configure
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/).

## Installation

If you are using virtual environments it's preferable to install JupyterLab
outside a virtual environment and add them later as kernels.

JupyterLab can be installed from `pip`:

```bash
pip3 instal jupyterlab
```

Then launch it with the following command:

```bash
jupyter-lab
```

If you are used to using tmux, you can run JupyterLab in the background with
the following command:

```bash
# launch tmux session in the background
tmux new -d -s jupyter "jupyter-lab --no-browser"

# attach to the session when you want to
tmux attach-session -t jupyter
```

## Attaching kernels

### Python virtual environment kernel

Kernels in JupyterLab are installed exactly like in regular Jupyter Notebooks.

We are going to create a first kernel based on Python 3 and we are going to
name it "Python (data-science)".

```bash
# create virtual env based on python3
virtualenv data-science -p python3

# activate it
source data-science/bin/activate

# install ipykernel within it
pip install ipykernel

# install the kernel into jupyter and give it a display name
python -m ipykernel install --user --name data-science --display-name "Python (data-science)"
```

### Julia kernel

You can use JupyterLab with other languages than Python, like Julia.

```julia
# install IJulia package (within Julia interpreter)
Pkg.add("IJulia")
```

Now reload JupyterLab and you should see that a new Julia kernel appeared.

## JupyterLab extensions

We can add widgets to JupyterLab, for instance:

- [Table of contents](https://github.com/jupyterlab/jupyterlab-toc):
  automatically generate a table of contents for your
  notebook, based on the markdown sections you wrote
- [Git](https://github.com/jupyterlab/jupyterlab-git):
  If working within a git repository, directly add & commit from
  the Jupyter Lab interface
- [Latex](https://github.com/jupyterlab/jupyterlab-latex):
  Write and compile Latex files from Jupyter Lab
- [Templates](https://pypi.org/project/jupyterlab-templates/):
  Lets you start a notebook from a template

Extensions can be installed from the command line by using the
`jupyter-labextension` command. All these commands must be run
outside of any virtual environment (use the same python in which
you installed jupyterlab).

### [Table of contents](https://github.com/jupyterlab/jupyterlab-toc)

This widget generates a table of content from your notebook.

Install it with:

```bash
jupyter-labextension install @jupyterlab/toc
```

You'll see a new tab appear, when you click on it, it'll show
the table of content. The sections are automatically created based on the
markdown sections. Note that you can either enable automatic numbering or not.

### [Git](https://github.com/jupyterlab/jupyterlab-git):

If you're tired of running back and forth between your terminal and Jupyter
Lab to commit your code, consider this extension that brings a Git interface
to Jupyter Lab.

Install it with:

```bash
pip3 install jupyterlab-git
jupyter-labextension install @jupyterlab/git
```

As usual, a new tab appears that lets you commit & push directly
from JupyterLab.

### [Latex](https://github.com/jupyterlab/jupyterlab-latex)

Use this plugin if you wish to use Jupyter Lab for compiling Latex documents.

Install it with:

```bash
pip3 install jupyterlab_latex
jupyter-labextension install @jupyterlab/latex
```

Now when you are editing a Latex file, a right click on the document will
show an option that lets you render the file to pdf.

### [Templates](https://pypi.org/project/jupyterlab-templates/)

Sometimes you might feel like you always copy paste the same lines of code
at the beginning of a new notebook. If you are, consider using notebook
templates with this plugin.

Install it with:

```bash
pip3 install jupyterlab_templates
jupyter labextension install jupyterlab_templates
jupyter serverextension enable --py jupyterlab_templates
```

Then you will see a new "Template" icon next to your kernels. If you click on
it, you will be asked which template you want to use, and it will create
a notebook based on this template. You can create your own templates by
saving notebooks in the template directory. You also get to choose which
directory the plugin will pick the templates from.
