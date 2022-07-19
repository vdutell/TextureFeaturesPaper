import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import scipy
from scipy.stats import kurtosis
from scipy.stats import entropy
from scipy.stats import moment
import pyrtools as pt

from scipy.signal import hilbert2

def srgb2lin(s):
    limit = 0.0404482362771082 
    s[s <= limit] = s[s <= limit]/12.92
    s[s > limit] = np.power(((s[s>limit] + 0.055) / 1.055), 2.4)
    return s

def lin2srgb(lin):
    limit = 0.0031308
    lin[lin > limit] = 1.055 * (np.power(lin[lin > limit], (1.0 / 2.4))) - 0.055
    lin[lin <= limit] = 12.92 * lin[lin <= limit]
    return lin

def get_texture(path, greyscale=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    if(greyscale):
        img = srgb2lin(img / 255.) #linearize
        img = (0.2126*img[:,:,0])+(0.7152*img[:,:,1])+(0.0722*img[:,:,2])#perceptually relavent norm
        img = lin2srgb(img)
    else:
        img = img / 255.
    return(np.array(img))


def calc_coarseness(img, log=True, normalizer=971000.9000868056): #949766.4334201389
    '''
    Calculate Contrast from Tamura et al 1967.
    
    Arguments:
        img (2d np array): Image to calculate contrast over
    Returns:
        f_cor: retrns single contrast value for entire image.
               If windowed=True, returns a list of contrast values for each window
    '''

    # We want 2 orientations, and 6 levels for 1080x1920 img
    filt = 'sp1_filters'
    img_pyr = pt.pyramids.SteerablePyramidSpace(img, order=1)
    pyr_height = int(len(img_pyr.pyr_size)/2 - 1)
    
    #placeholders for max values and scales at which max values ocurred
    max_act = np.zeros_like(img_pyr.pyr_coeffs[(0,0)])-1
    max_act_scale = np.zeros_like(img_pyr.pyr_coeffs[(0,0)]) - 1
    #loop through each scale (+1 in places because python is zero-indexed)
    for scale in range(pyr_height):
        #calculate the absolute value of the horiz & vert filts at this scale, and normalize
        vert_filt_img = np.abs(img_pyr.pyr_coeffs[(scale,0)])/(2**(2*(scale+1)))
        horiz_filt_img = np.abs(img_pyr.pyr_coeffs[(scale,1)])/(2**(2*(scale+1)))
        #compute the magnitude of maximum filter response (l2 norm of horiz and vert filters)
        scalemag = np.sqrt(img_pyr.pyr_coeffs[(scale,0)]**2 + img_pyr.pyr_coeffs[(scale,1)]**2)
        #reshape the filter image to full size for higher levels 
        if(max_act.shape != scalemag.shape):
            scalemag = cv2.resize(scalemag, dsize=max_act.shape[::-1], interpolation=cv2.INTER_CUBIC) #INTER_LINEAR is faster
        #create mask where new scale max is larger
        max_act_mask = np.argmax(np.stack((max_act, scalemag),axis=-1),axis=-1).astype(bool)
        max_act_scale[max_act_mask] = scale+1
        #replace global max with current max if current is larger
        max_act = np.max(np.stack((max_act, scalemag),axis=-1), axis=-1)

    #f_crs = np.mean(max_act_scale) #equivalent of log scaling
    if(log):
        f_crs = np.mean(10**max_act_scale) / normalizer                    
    else:
        f_crs = np.mean(max_act_scale)

    return(f_crs)


# def calc_contrast_rms(img, log=True, normalizer=10.): 
#     '''
#     Calculate Contrast
    
#     Arguments:
#         img (2d np array): Image to calculate contrast over
#     Returns:
#         f_con: retrns single contrast value for entire image.
#     '''
    
#     dmean = (img - np.mean(img))**2 #squared difference
#     dmean = np.sqrt(np.mean(dmean)) #sqrt of mean
        
# #     if(log):
# #         dmean = np.log10(dmean)
# #     dmean = dmean / normalizer

#     return(dmean)


def calc_contrast_peli(img, delta=0.01, normalizer = 9.04993072433997): #8.655381330712304
    
    #Laplacian Pyarmid
    pyr = pt.pyramids.LaplacianPyramid(img)

    num_bands = len(pyr.pyr_size) #exclude low and high
    mean_contrast = []
    
    #loop through each scale (+1 in places because python is zero-indexed)
    for i in range(0, num_bands-1):
        level_recon = pyr.recon_pyr(levels=list(range(i+1, num_bands)));
        level_img = pyr.pyr_coeffs[i,0]
        level_img_resized = cv2.resize(level_img, dsize=pyr.pyr_coeffs[0,0].shape[::-1], interpolation=cv2.INTER_CUBIC) #INTER_LINEAR is faster
        #all 1920x1080

        #Local Contrast Image
        contrast_img = level_img_resized / (level_recon + delta) #small offset to make sure we dont divide by zero
        #magnitude
        mag_contrast_img = np.abs(contrast_img)

        mean_contrast.append(np.mean(mag_contrast_img))

    f_con = np.mean(mean_contrast)
    
    f_con = f_con / normalizer
    
    return(f_con)
    
# def calc_contrast_mch(img,log=True, normalizer=10.): 
#     '''
#     Calculate Contrast from Tamura et al 1967.
    
#     Arguments:
#         img (2 or 3d np array): Image to calculate contrast over
#     Returns:
#         f_con: retrns single contrast value for entire image.
#                If windowed=True, returns a list of contrast values for each window
#     '''
    
#     imax = np.max(img)
#     imin = np.min(img)
#     f_con = (imax - imin)/(imax + imin)
#     if(log):
#         f_con = 10**f_con 
        
#     f_con = f_con / normalizer

#     return(f_con)


# def calc_contrast_tamura(img, return_std=True): 
#     '''
#     Calculate Contrast from Tamura et al 1967.
    
#     Arguments:
#         img (2 or 3d np array): Image to calculate contrast over
#     Returns:
#         f_con: retrns single contrast value for entire image.
#                If windowed=True, returns a list of contrast values for each window
#     '''
    
#     def contrast_on_img(im):
#         var = np.var(im)
#         #std = np.std(im)
#         #moment_4th = moment(im,moment=4)
#         #kurt = moment_4th/(std**2)
#         kurt = kurtosis(im, fisher=False)
#         fcon = var*(kurt**(4))
#         #fcon = var/(kurt**(-4))
#         return(fcon)
    
#     f_con = contrast_on_img(img.flatten())
#     f_con = np.log10(f_con)
#     return(f_con)


def calc_directionality(img, normalizer=34964.119837117534 , num_bins=100): #35321.24248025348
    #wit log normlizer is 10.002331268313245
    #without log: 1.6376529477002038
    '''
    Calculate directionality of a texture:
    Parameters:
        img (2d Numpy array): Image to have directionality calculated
    Returns:
        f_dir (float): Directionality Coefficient
    '''

    ## Use Pyramid Decomposition
    filt = 'sp1_filters'
    img_pyr = pt.pyramids.SteerablePyramidSpace(img, order=1)
    pyr_height = int(len(img_pyr.pyr_size)/2 - 1)
    level_entropies = []
    
    for pyrlevel in range(pyr_height):
        vert_filter = img_pyr.pyr_coeffs[(pyrlevel,0)]
        horiz_filter = img_pyr.pyr_coeffs[(pyrlevel,1)]

        #calculate magnitudes and angles
        mag_g = np.sqrt((vert_filter)**2+(horiz_filter)**2)
        ang_g = np.arctan2(vert_filter, horiz_filter) 
        mag_thresh = np.mean(mag_g)-np.std(mag_g)
        ang_g_masked = ang_g[np.where(mag_g > mag_thresh)]

        #normalize by area to get pdf for entropy calc
        ang_pdf, _ = np.histogram(ang_g_masked, density=True, bins=num_bins)
        ang_pdf = ang_pdf[ang_pdf != 0] #entropy can't be calculated with zero bins
        #calculate entropy on Angle Distribution
        level_entropy = entropy(ang_pdf)
        level_entropies.append(level_entropy)
        
    f_dir = 10**(np.min(level_entropies))

    f_dir = 1 - (f_dir / normalizer)
    
    return f_dir

def calc_linelikeness(img, window_size_min = 32, window_size_max=128, plot_windows=False, exclude_uniform=True, num_windows=100, log=True, normalizer=9.323480929896597): #9.258866094196945
    '''
    Calculate Line-Likeness of a texture by looking at the entropy of directionality
    over smaller windows in the image. This may indicate local lines.
    '''
    xdims,ydims = img.shape
    directionality_list = []
    for exp in range(int(np.log2(window_size_min)), int(np.log2(window_size_max))+1):
        window_size = 2**exp
        directionality_window_list = []
        for _ in range(num_windows):
            x = np.random.randint(0,xdims-window_size)
            y = np.random.randint(0,ydims-window_size)
            window = img[x:x+window_size, y:y+window_size]

            if(exclude_uniform and (np.var(window)<1e-5)):
                directionality_window_list.append(0)
            else:
                directionality_window_list.append(calc_directionality(window))

        directionality_list.append(np.mean(directionality_window_list))
    
    f_ll = np.max(directionality_list)
    
    if log:
        f_ll = (10**f_ll) / normalizer
    
    return(f_ll)


def calc_roughness_phasecon(img, pyramid=True, log=True, norm_min=0.8603382084936996, norm_max = 1.6127322599012415 ): #norm_min=1.2699284050987591, norm_max=1.6428554597153535
    '''
    Calculate roughness of a texture (based on phase congruency)
    Parameters:
        img (2d Numpy array): Image to have roughness calculated
    Returns:
        f_dir (float): Roughness Coefficient
    '''   
    
    if(pyramid):
        order = 5 #3
        img_pyr = pt.pyramids.SteerablePyramidFreq(img, order=order, is_complex=True)
        pyr_height = int((len(img_pyr.pyr_size)-2)/(order+1))
        pc_vals = []
        
        for pyrlevel in range(pyr_height):
            #grab filter image
            vert_filter = img_pyr.pyr_coeffs[(pyrlevel,0)]
            horiz_filter = img_pyr.pyr_coeffs[(pyrlevel,3)]
            diag1_filter = img_pyr.pyr_coeffs[(pyrlevel,1)]
            diag2_filter = img_pyr.pyr_coeffs[(pyrlevel,2)]

            #append magnitude of real and imaginary components to 
            pc_vals.append(np.mean(np.sqrt(np.real(vert_filter)**2 + np.imag(vert_filter)**2)))
            pc_vals.append(np.mean(np.sqrt(np.real(horiz_filter)**2 + np.imag(horiz_filter)**2)))
            pc_vals.append(np.mean(np.sqrt(np.real(diag1_filter)**2 + np.imag(diag1_filter)**2)))
            pc_vals.append(np.mean(np.sqrt(np.real(diag2_filter)**2 + np.imag(diag2_filter)**2)))
            
            if(order==5):
                diag3_filter = img_pyr.pyr_coeffs[(pyrlevel,4)]
                diag4_filter = img_pyr.pyr_coeffs[(pyrlevel,5)]
                pc_vals.append(np.mean(np.sqrt(np.real(diag3_filter)**2 + np.imag(diag3_filter)**2)))
                pc_vals.append(np.mean(np.sqrt(np.real(diag4_filter)**2 + np.imag(diag4_filter)**2)))

        if(log):
            r = (np.mean(np.log10(np.array(pc_vals))) + norm_min ) / norm_max
        else:
            r = np.mean(pc_vals)
            
    else:
        #cheap calculation with 2D image (only vertical and horizontal)
        im = img - np.mean(img)
        hil = hilbert2(im)
        energy2d = np.real(np.sqrt(hil**2 + im**2))
        if(log):
            energy2d = np.log10(energy2d) / normalizer
        r = np.mean(energy2d)
    
    return(r)


def calc_regularity(img, window_size_min = 32, window_size_max=256, exclude_uniform=True, num_windows=100, log=True, normalizer=0.9806178069261438):
    ''' 
    Calculate regualrity of texture (based on variation of coarseness and contrast)
    Parameters:
        img (2d Numpy array): Image to have directionality calculated
        window_size(int): Edge size of the window taken on image

    Returns:
        f_dir (float): Directionality Coefficient
    '''
    
    xdims,ydims = img.shape
    std_cor_levels = []
    std_con_levels = []
    std_dir_levels = []
    std_rgh_levels = []
    #loop over window sizes
    for exp in range(int(np.log2(window_size_min)), int(np.log2(window_size_max))+1):
        window_size = 2**exp
        coarseness_list = []
        contrast_list = []
        directionality_list = []
        roughness_list = []
        
        for _ in range(num_windows):
            x = np.random.randint(0,xdims-window_size)
            y = np.random.randint(0,ydims-window_size)
            window = img[x:x+window_size, y:y+window_size]

            if(exclude_uniform and (np.var(window)<1e-5)):
                pass
            else:
                if(window_size > 32):
                    coarseness_list.append(calc_coarseness(window))
                    roughness_list.append(calc_roughness_phasecon(window))
                contrast_list.append(calc_contrast_peli(window))
                directionality_list.append(calc_directionality(window))

        #append std of this window size to list
        if(window_size > 32):
            std_cor_levels.append(np.std(coarseness_list))
            std_rgh_levels.append(np.std(roughness_list))
        std_con_levels.append(np.std(contrast_list))
        std_dir_levels.append(np.std(directionality_list))
        
    #take average of stds for each window size (this way small windows arent counted more)
    std_cor_levels = np.min(std_cor_levels)
    std_con_levels = np.min(std_con_levels)
    std_dir_levels = np.min(std_dir_levels)
    std_rgh_levels = np.min(std_rgh_levels)
    
    f_reg = 1 - (std_cor_levels + std_con_levels + std_dir_levels + std_rgh_levels)
    return(f_reg)

def randomize_phase(img, norm=True, scrambled=False):
    '''
    Randomize the phase in an image
    Arguments:
        img (2d Numpy Array): image to have phase scrambled
        scrambled (bool): Reconstruct with scrambled phase instead of uniform random
        norm (bool): scale reconstructed image to [0,1]?
    Returns:
        recon (2d Numpy Array): image with phase scrambled
    '''
    fft = np.fft.fft2(img)
    mag = np.abs(fft)
    if(scrambled):
        phase = np.angle(fft)
        np.random.shuffle(phase)
        phase = phase.T
        np.random.shuffle(phase)
        phase = phase.T 
    else:
        phase = np.random.uniform(-np.pi,np.pi,size=(fft.shape))
    newfft = mag*np.exp(1j*phase)
    recon = np.real(np.fft.ifft2(newfft))
    if(norm):
        recon = recon / np.max(recon)
    return(recon)


def calc_roughness_recon(img, scrambled=False, verbose=False):
    '''
    Calculate roughness of a texture (based on coarseness,contrast,directionality,regularity)
    Parameters:
        img (2d Numpy array): Image to have roughness calculated
        scrambled (bool): Reconstruct with scrambled phase instead of uniform random
    Returns:
        f_dir (float): Roughness Coefficient
    '''   
    #calc stats for orininal img
    ori_cor = calc_coarseness(img)
    ori_con = calc_contrast_peli(img)
    #ori_dir = calc_directionality(img)
    #ori_reg = calc_regularity(img)
    #ori_lln = calc_linelikeness(img)
    #calculate phase randomized image
    scr_img = randomize_phase(img, scrambled=scrambled)
    #calculate state for phase randomized image
    scr_cor = calc_coarseness(scr_img)
    scr_con = calc_contrast_peli(scr_img)
    #scr_dir = calc_directionality(scr_img)
    #scr_reg = calc_regularity(scr_img)
    #scr_lln = calc_linelikeness(scr_img)
    
    if(verbose):
        print(f'Corseness vals are:{ori_cor:0.4f},{scr_cor:0.4f}')
        print(f'Contrast vals are:{ori_con:0.4f},{scr_con:0.4f}')
        print(f'Directionality vals are:{ori_dir:0.4f},{scr_dir:0.4f}')
        print(f'Regularity vals are:{ori_reg:0.4f},{scr_reg:0.4f}')
        print(f'LineLikeness vals are:{ori_lln:0.4f},{scr_lln:0.4f}')
    
    #A rough image should retain its statistical properties even when phase is scrambled
    #if the statistical properties change - this is a deviation from roughness
    #normalize the differences by the maximum magnitude
    diff_cor = np.abs(ori_cor - scr_cor) / np.max(np.abs((ori_cor, scr_cor)))
    diff_con = np.abs(ori_con - scr_con) / np.max(np.abs((ori_con, scr_con)))
    #diff_dir = np.abs(ori_dir - scr_dir) / np.max(np.abs((ori_dir, scr_dir)))
    #diff_reg = np.abs(ori_reg - scr_reg) / np.max(np.abs((ori_reg, scr_reg)))    
    #diff_lln = np.abs(ori_lln - scr_lln) / np.max(np.abs((ori_lln, scr_lln)))
    
    if(verbose):
        print(f'Corseness Difference is:{diff_cor:0.4f}')
        print(f'Contrast Difference is:{diff_con:0.4f}')
        #print(f'Directionality Difference is:{diff_dir:0.4f}')
        #print(f'Regularity Difference is:{diff_reg:0.4f}')
        #print(f'LineLikeness Difference is:{diff_lln:0.4f}')

    #this is where the other stats not being normalized [0-1] is a problem.
    f_rough = 1 - ((diff_cor + diff_con) / 2)
    
    return(f_rough)

def calc_topside_con_ratio(folder, fname, met=False):
    '''
    Caclulate the ratio of side vs top contrast
    folder: name of folder holding textures
    fname: filename of current textures
    '''
    
    #contrast ratio (top side)
    texnum = int(str(fname).replace('Tex_','').replace('_orig.jpg','').replace('_metm.jpg',''))
    #numbers below 20 are images
    if(texnum<20):
        f_topside_con_ratio = 1
    #numbers 20 and above are pbr textures
    else:
        #if it's an even number its the side lit version
        if(texnum%2 ==0):
            if(met):
                side_fname = os.path.join(folder,f'Tex_{str(texnum).zfill(3)}_metm.jpg')
                top_fname = os.path.join(folder,f'Tex_{str(texnum+1).zfill(3)}_metm.jpg')
            else:
                side_fname = os.path.join(folder,f'Tex_{str(texnum).zfill(3)}_orig.jpg')
                top_fname = os.path.join(folder,f'Tex_{str(texnum+1).zfill(3)}_orig.jpg')    
        #if it's an odd number its the top lit version
        else:
            if(met):
                side_fname = os.path.join(folder,f'Tex_{str(texnum-1).zfill(3)}_metm.jpg')
                top_fname = os.path.join(folder,f'Tex_{str(texnum).zfill(3)}_metm.jpg')   
            else:
                side_fname = os.path.join(folder,f'Tex_{str(texnum-1).zfill(3)}_orig.jpg')
                top_fname = os.path.join(folder,f'Tex_{str(texnum).zfill(3)}_orig.jpg')    


        side_contrast = calc_contrast_peli(get_texture(side_fname, greyscale=True))
        top_contrast = calc_contrast_peli(get_texture(top_fname, greyscale=True))
        f_topside_con_ratio = side_contrast / top_contrast
    return(f_topside_con_ratio)

def azimuthalAverage(image, center=None, bin_in_log=False):
    """      
    image - The 2D image (2d power spectrum)
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    num_bins = np.min(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    #ASSUME HERE THAT MAX FREQUENCY IS EQUAL ON BOTH AXES & GRANULARITY VARIES ***
    normalized = ((x-center[0])/np.max(x),(y-center[1])/np.max(y))
    r = np.hypot(normalized[0], normalized[1])
    #don't calculate corners
    keep_circle = np.where(r<=np.max(y))
    r = r[keep_circle]
    image = image[keep_circle]

    # number of bins should be equivalent to the number of bins along the shortest axis of the image.
    if(bin_in_log):
        bin_edges = np.histogram_bin_edges(np.log(r), num_bins)
        bin_edges = np.exp(bin_edges)
    else:
        bin_edges = np.histogram_bin_edges(r,num_bins)
    
    r_binned = np.digitize(r, bin_edges)
    binmean = np.zeros(num_bins)
    for i in range(num_bins):
        #if(len(r_binned[r_binned==i+1])>0):
        binmean[i] = np.mean(image[np.where(r_binned==i+1)])
        #else:
        #    binmean[i] = 0
    bin_centers = bin_edges[:-1] + ((bin_edges[1]-bin_edges[0])/2)
    bin_centers = (bin_centers/np.max(bin_centers))

    return(binmean, bin_centers)


def fit_alpha_powerspec(img):
    print('*',end='')
    
    def calc_power_spec_slope(img):
        ps = np.abs(np.fft.fftshift(np.fft.fft2(img-np.mean(img))))**2
        avg = azimuthalAverage(ps)
        return(avg)

    def feq(cpd, a, alpha):
        amp = a / (cpd**alpha)
        return(amp)

    bin_means, bin_centers = calc_power_spec_slope(img)
    log_means = np.log10(bin_means) #fit in logspace
    [a_fit, alpha_fit], cov = scipy.optimize.curve_fit(feq, bin_centers, log_means)
    
    return(alpha_fit)