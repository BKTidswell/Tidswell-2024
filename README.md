# Code for (Title of this Paper)

Welcome to this Github respository for (Title of this Paper). If you are here, I assume you're interested in doing a deep dive into the code that I wrote to turn the positional data of fish into schooling kinematics. To that end this guide aims to walk you through running all the code from start to finish, and to explain the purpose the code files here. If you run into any issues with it feel free to raise an issue here or email me at b.k.tidswell@gmail.com. 

As a note, this code was written to work on a Mac, so you may need to set up some things differently if you are using a Windows and Linux computer, particularly with the Conda enviornment, as well as some of the graphing functions. With that said, let's get started.

## A Tour of the Files

In this folder there are a variety of subfolders that contain the files needed to go from the raw traces of points from DeepLabCut all the way to the finished graphs and statistics used in the paper. These are listed here in the order they are used the in data processing pipeline.

2Dto3D/ - This folder contains the files and code for taking the digitized points from Ventral 1 (V1) and Ventral 1 (V2) cameras and creating 3D points
      V1 CSVs/ - Contains the V1 positional data
      V2 CSVs/ - Contains the V1 positional data
      DLT Coefs/ - Contains the DLT Coeficents needed to turn the two 2D views into one 3D set of points
      DLT_Converter.py - This script takes the data from V1 CSVs/ and V2 CSVs/ and creates the 3D points, putting them in Final 3D/
      DLT_Plotter.py - This script take the 3D Points in Final 3D/ and created 3D plots of them, putting the HTML files in Saved 3D Plots/
      Final 3D/ - Contains the 3D positions of fish
      Saved 3D Plots/ - Contains 3D plots of the 3D positions of fish

## Installing Environment


## 2D to 3D Data


## 3D Data to School Kinematics CSV


## School Kinematics CSV to Paper Graphs and Stats
