# Projects in Julia 

Even for a personal project, we recommend to make a simple Julia package. This is a simple tutorial to help you coding an app in Julia.

## Prerequisite

You need to have [Julia](https://julialang.org/downloads/) installed and a [GitHub](https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account) account.

## Creating the Julia Package

We'll be using [PkgSkeleton.jl](https://github.com/tpapp/PkgSkeleton.jl) which allows to simplify the creation of packages. First check your git configuration (as it will be used to create the package) with:
```
git config --list
```
You should see your `user.name`, your `user.email` and `github.user`, if not then use for example:
```
git config --global user.name "firstname lastname"
git config --global user.email "bla.bla@domain.ext"
git config --global github.user "ghuser"
```

Now, I describe the steps used to make a package called [KalmanAD.jl](https://github.com/mlelarge/KalmanAD.jl), you need to replace `KalmanAD.jl` by the name of your package and `mlelarge` by your `github.user`! This particular package is irrelevant but since naming (with extension `.jl` for example) can be a little bit tricky, if you have any doubt, you can have a look at the package to see how things are organized...

Move to the folder where you want to create your package: we will use Julia to create a folder `KalmanAD.jl` with the right structure. For this, start Julia and enter the package installation mode by typing `]` then run (to exit the `pkg>` mode just type on Backspace)
```
pkg> add PkgSkeleton
julia> using PkgSkeleton
julia> PkgSkeleton.generate("KalmanAD.jl")
```
Now, when your exit Julia, you should have a folder `KalmanAD.jl` with some files and folders in it.

## Connecting to GitHub

Go on GitHub.com and create a repository with the same name: `KalmanAD.jl` by following [these steps](https://docs.github.com/en/get-started/quickstart/create-a-repo) (make it public).

On your computer, inside your `KalmanAD.jl` run the following commands:
```
git add . && git commit -m "initial commit"
git remote add origin git@github.com:mlelarge/KalmanAD.jl.git
git branch -M main
git push -u origin main
```
The first command add the files created by `PkgSkeleton` and commit them. The last 3 commands are connecting your git repo to GitHub (of course you need to replace `mlelarge` and `KalmanAD.jl` by the appropriate values). Now, you should see on your GitHub account the repository you created on your computer.

## Start coding

You have 3 sub-folders in your Package `docs`, `src` and `test`. Your code should be in the `src` subfolder. In particular, in the `src` folder there should be a file with the same name as the package, (i.e. `KalmanAD.jl` here) and this file contains:
```
module KalmanAD
end #module
```
This file will need to be modified as it will define the `module` of your package.

Now, when you start coding, you will use other Julia packages. For example, you can see in [KalmanAD.jl](https://github.com/mlelarge/KalmanAD.jl/blob/main/src/KalmanAD.jl) that I am using the package `LinearAlgebra`. So I need to add it as a dependency of my own package (a bit like a virtual env in python), to do so you need to run Julia and activate the environment of your package with the command:
`julia --project`. Now if you type `]`, you should see:
```
(KalmanAD) pkg>
```
and you can now add the packages you need, for example:
```
(KalmanAD) pkg> add LinearAlgebra
```
as a result this will modify (automatically) the files `Project.toml` and `Manifest.toml`. Next time, you commit, do not forget to add these files.

## Start testing

TBD

## Start documenting

TBD