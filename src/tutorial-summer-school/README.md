# SeaPearl Tutorial

This tutorial is designed to introduce you to SeaPearl. SeaPearl is a Julia package that allows researchers to build reinforcement learning agents to learn value selection heuristics. Strongly inspired by [MiniCP](http://www.minicp.org/) the package is designed to be easy to use and to allow researchers to quickly prototype new ideas. The package is also designed to be easy to extend, new algorithms, and new heuristics.

## Pre-requisites and installation

To speed things up during this tutorial, you should have some software + packages installed. This section explains what to install and how.

### Julia
You should have julia 1.8 installed. If using a linux distribution, here is how to do it:
    
```bash
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.0-linux-x86_64.tar.gz
    tar -xvzf julia-1.8.0-linux-x86_64.tar.gz
    sudo mv julia-1.8.0 /opt/
    sudo ln -s /opt/julia-1.8.0/bin/julia /usr/local/bin/julia
    echo "Julia 1.8.0 has been installed"
```
If using Mac or windows, head over to the [releases page](https://julialang.org/downloads/oldreleases/) of the julia website, find version 1.8.0 and download the appropriate installer. Follow the usual steps to install the software.

### SeaPearlZoo.jl

The tutorial is located inside [SeaPearlZoo.jl](https://github.com/corail-research/SeaPearlZoo.jl). SeaPearlZoo contains many examples of SeaPearl usage. Head over to the [project's github page](https://github.com/corail-research/SeaPearlZoo.jl) and git clone the repository. 

```bash
git clone https://github.com/corail-research/SeaPearlZoo.jl.git
```

### Jupyter Notebooks

The tutorial is written in the form of Jupyter notebooks. To install Jupyter, you can use the following command:
```bash
julia -e 'using Pkg; Pkg.add("IJulia")'
```

## Enjoy the tutorial!

With these in hand, you should be good to go! Enjoy!