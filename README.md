# Code for (Title of this Paper)

Welcome to this Github respository for (Title of this Paper). If you are here, I assume you're interested in doing a deep dive into the code that I wrote to turn the positional data of fish into schooling kinematics. To that end this guide aims to walk you through running all the code from start to finish, and to explain the purpose the code files here. If you run into any issues with it feel free to raise an issue here or email me at b.k.tidswell@gmail.com. 

As a note, this code was written to work on a Mac, so you may need to set up some things differently if you are using a Windows and Linux computer, particularly with the Conda enviornment, as well as some of the graphing functions. With that said, let's get started.

## Installing the Environment

I have included a conda environement file (DLC_M1.yml) to use to set up the environement that I used to run all of this code. This does include the DeepLabCut libraries, as well as tensorflow, which are not needed to run any of the code in this repository, but are included becasue that is the environment I used. 

With Conda installed, simply run `conda env create -f DLC_M1.yml` to create the environment, and then `conda activate DLC_M1` once it is finished to use it.
 
## 2D to 3D Data


## 3D Data to School Kinematics CSV


## School Kinematics CSV to Paper Graphs and Stats
