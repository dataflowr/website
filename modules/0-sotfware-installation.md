# Setup

## Running the notebooks locally

To run the [notebooks](https://github.com/dataflowr/notebooks) locally, we recommend the following procedure:

- First clone the GitHub repository containing the notebooks. The following command will create a directory `notebooks` with all the files from the repository inside:
```
$ git clone https://github.com/dataflowr/notebooks.git
```

- Then, create a [virtual environment](https://docs.python.org/3/tutorial/venv.html): the following command will create a directory `dldiy` and also create directories inside it (so you might want to create this directory inside `/notebooks`)
```
$ python3 -m venv dldiy
```
- Activate the virtual environment:
```
$ source dldiy/bin/activate
```
- In order to be able to [use this virtual environment with your jupyter notebook](https://anbasile.github.io/posts/2017-06-25-jupyter-venv/), you need to add a kernel. Inside your environment, install first `ipykernel`
```
(dldiy)$ pip install ipykernel
(dldiy)$ ipython kernel install --user --name=dldiy
```
- Now, you need all the relevant packages in your virtual environment. 
```
(dldiy)$ cd notebooks
(dldiy)/notebooks$ pip install -r requirements.txt
```
- You ae all set! If you launch `jupyter notebook`, you should be able to change the kernel to `dldiy`.

### tl;dr

```
$ git clone https://github.com/dataflowr/notebooks.git
$ python3 -m venv dldiy
$ source dldiy/bin/activate
(dldiy)$ pip install ipykernel
(dldiy)$ ipython kernel install --user --name=dldiy
(dldiy)$ cd notebooks
(dldiy)/notebooks$ pip install -r requirements.txt
(dldiy)/notebooks$ jupyter notebook
```
