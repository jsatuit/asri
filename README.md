# asri – Code for Aperture Synthesis Radar Imaging
Code to simulate measurements of an aperture synthesis radar and analyse this data.
The focus of the program is at EISCAT 3D.

This repository started with the replication code for the article [Radar imaging with EISCAT 3D](https://doi.org/10.5194/angeo-39-119-2021).
The original code can be downloaded from [Dataverse](https://doi.org/10.18710/QRDET2).

DOI Published paper: 10.5194/angeo-39-119-2021
DOI of originally published code: 10.18710/QRDET2
Author: Johann Stamm

## License

The source code of `asri` is distributed under the terms of the [MIT-no attribution licence](https://spdx.org/licenses/MIT-0.html).
The licence of the images varies per file. For more information, see [programdata/licence.md](programdata/licence.md).



## Usage
Doing imaging or plotting the radar layouts make use of the file radaro3.py. 
This requires that the variable 'fillagresti' in this file is set. 
The variable is a string that defines where results and intermediate calculations are saved. 
Since these will take up space, this folder should be on a hard drive with enough capacity. 
Additionally, most of the imaging functions need matrix inversions of large matrices. 
Therefore a computer with large memory is needed. 
Most of the funcions will work with ~8 GB RAM, but if the imaging resolution is above 20x20 pixels, more is needed. 
This also accounts for plotting the radar layouts.


To create and save measurements you will have to import radar.py. 
But before you have to change the path for saving directory.


### Code description
All functions are written in Python version 3 and have been used with version 3.6.9. 
The functions depend on the following packages: numpy, Matplotlib (pyplot), imageio, and h5py.

When using the code, there might be some cases where expected figures do not appear.
A reason for this can be that the function for showing plots is not called. 
This can be done by adding a line "show_plots()" at the bottom of the script. 
This calls matplotlib.pyplot.show() to open a window with all plots created in the run. 
If using spyder, this should not be a problem because spyder shows created plots immedeately.

The warning on the normalizing of the measurements is because the code used originally is located on on unavailable server. 
Therefore this function may not work as expected.

### Replicate paper figures (and data)
Most likely, most of the users of this code want to reproduce the figures in the article.
They were created inf following manner:

For figure 2, this is done by running the file "radarl.py" as a script (not importing)
For figures 4 and 5, this is done by running krysskorr_normford.py. 
Figure 7 is plotted when running radaro3.py.  

Plotting comparison of images as in figure 8 is doing in recovering.py. For exampleone can call
reccalc(layouts,'nordlys',0.05,100e3,1,[10,20],['LS','TSVD','capon'],[None,2e-2,None],False,True,False)


The code might not work in the first try. Sometimes certain folders are missing, see requirements above. 
These have then to be created manually (and can be let empty, they will be filled). 
Additionally, the original code is not currently available and this code does not represent the final stage.

This means that we use the original image 'nordlys', which is the auroral image. 
The measurements have relative 5 % noise added onto them. 
The range to the target is 100 km, 
The radar HPBW is 1°. The inages will be calculated for all assumed resolutions between 10x10 and 20x20 pixels. 
The recovering algorithms used are 'LS' (least squares), 'TSVD' (truncated SVD), and 'capon'. 
Another possibility is 'MF' (Matched filter). The next argument contains parameters the methods are using. Only TSVD can use this. 
Methodparameters set for the other techniques are ignored. 
Here TSVD gets the parameter 2e-1 which means that singular values below 0,02 times the largest singular value are ignored when inverting the theory matrix. 
the next four possible arguments are for using farfield (may not work properly), 
showres for showing the result after computing (True, False, or 'save' for saving them in the folder fillagresti), 
showsds for showing the standard deviation of the results (these are many plots), 
and normcomp for comparing the original image to the normalized or unnormalized reconstruction (Not shown, default and recommended is False).

This function may also be called with the 'methods' parameter as 'svddiag' instead of a list. 
Then the reconstructions will look like figures 9-12. The argument afterwards must then be a number which tells where to truncate the singular values. 
The two parameters for the Tikhonov regularization are set in radaro3.py as trp1 and trp2. 

All calculations are then done for all layouts set, and all methods and resolutions in the range specified. 
This will take time to calculate. The results are plotted and the uncertainties saved in a hdf5 file (if stated), and their uncertainties are saved for every run.

By running the lines on the bottom of recovering.py the uncertainties can be plotted. 
As a default, suncertainties of the last set of imaging is plotted.