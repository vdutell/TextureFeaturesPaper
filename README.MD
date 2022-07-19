## Texture Feature Analysis

This repository contains the packaged codebase used for the texture feature analysis portion of Brown et al 2022, 'Efficient Dataflow Modeling of Peripheral Encoding in the Human Visual System'. Statistics computed are adapted from Tamura, Mori & Yamawaki 1978.

Included is the code to calculate the 6 statisticsal categories (Coarseness, Contrast, Directionality, Regularity, Line-Likeness, Roughness) for texture images. 


Dependencies:  
    - Pyrtools (Python Version of PyrTools) https://pyrtools.readthedocs.io/en/latest   
    - Multiprocessing https://docs.python.org/3/library/multiprocessing.html  
    - Open CV (CV2)  
    - Pandas  

The statistical calculation code is contained in texture_features.py : The fucntions inside this file can be used to calculate these statistical features for any color (three channel) image! Note these functions take a 'normalizer' value, which you likely want to set to normalizer=1 for an image different than what is used in the paper. The normalizer is calculated speficially for this dataset to normalize the distribution of the image set used in our experiment.

Texture_Distributions.ipynb is a jupyter/ipython notebook that calculates the texture features for the texture set used in the paper, which have been tested psychophysically.  
    - To run the notebook and reproduce the paper results, you will need to download this texture database as well and follow the directions in the notebook to use them.   
    - You can also use this notebook as an example of how to call the texture_features.py functions on arbitrary images. 
    
The Notebook uses python's multiprocessing package to call the texture_features.py functions, as calculating many of these features is quite computationally expensive. You'll need to install this as well as cv2 and pandas, as well as Pyrtools, used to calculated Laplacian Pyramids for Peli Contrast.


Questions regarding this texture analysis can be directed to vasha@berkeley.edu
