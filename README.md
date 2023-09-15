# Deep Learning Do It Yourself

This site collects resources to learn Deep Learning in the form of Modules
available through the sidebar on the left. As a student, you can walk
through the modules at your own pace and interact with others thanks
to the associated digital platforms. Then we hope you'll become a
contributor by adding modules to this site!

## Setup to run the website locally

1. Install [Julia](https://julialang.org/downloads/) (make sure to 'add to PATH' so you can use step 2 without having to type the full path)
2. Launch Julia from the command line:

    ``` text
    julia
    ```

3. Install the required packages:

   ``` julia
   using Pkg
   Pkg.add("Franklin")
   Pkg.add("JSON")
   ```

4. Serve the website (on [localhost:8000](http://localhost:8000)):

    ``` julia
    using Franklin
    serve()
    ```

Note: steps 3 and 4 are automated in the [Makefile](./Makefile),
so you can just run `make install` and `make serve`.
