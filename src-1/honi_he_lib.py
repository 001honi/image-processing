"""
@Author: Honi, Selahaddin (honis@kaist.ac.kr)
@Date: 2022 March
"""
import cv2 as cv 
import numpy as np 
from tqdm import tqdm

class HE():
    """
    Histogram Equalization using procedure
        [0] set max intensity level initially by HE(L=256) 
        [1] set an input image (BGR) by setInput(img_BGR) method
        [2] perform the selected algorithm by apply(mode={0,1,2},kernel_size,target_hist)
        [3] get output image by getOutput()
        [4] get original and HE histograms by instance attributes he.r_pmf and he.s_pmf respectively
    """
    modes = {
        0:'Histogram-Equalization',
        1:'Local Histogram-Equalization',
        2:'Histogram Specification'
    }

    def __init__(self,L=256):
        self.L = L
        self.levels = list(range(L)) # intensity levels
        self.Y     = None # original-image intensity map 
        self.Cb    = None 
        self.Cr    = None
        self.Y_he  = None # he-image intensity map 
        self.r_pmf = None # original-image histogram
        self.s_pmf = None # he-image histogram
        self.tr    = None # transformation function


    def setInput(self,img_BGR):
        # Convert input image into YCrCb color-space and set Y intensity map
        img_YCrCb = cv.cvtColor(img_BGR,cv.COLOR_BGR2YCR_CB)
        (Y,Cr,Cb) = cv.split(img_YCrCb)

        self.Y = Y 
        self.Cb = Cb
        self.Cr = Cr

    
    def getOutput(self):
        img_YCrCb = cv.merge([self.Y_he,self.Cr,self.Cb])
        img_BGR = cv.cvtColor(img_YCrCb, cv.COLOR_YCR_CB2BGR)
        return img_BGR


    def apply(self,mode=0,kernel_size=7,target_hist=None):
        # Check validity
        valid = 1
        if self.Y is None: 
            valid = 0; print('(!) Set input image')
        if mode == 1 and kernel_size > min(self.Y.shape):
            valid = 0; print('(!) Kernel size cannot be greater than image dimensions in Local-HE')
        if mode == 2 and target_hist == None:
            valid = 0; print('(!) Set target histogram for Histogram Specification')
        if mode == 2 and len(target_hist) != self.L:
            valid = 0; print('(!) Set appropriate target histogram for Histogram Specification')

        if valid:
            print(HE.modes[mode] + ' is being applied...')
        else:
            exit()


        # Algorithm selection
        if mode == 0: # Standard HE
            r_pmf,tr,Y_he = self._apply(self.Y)
    
        elif mode == 1: # Local HE
            # Window locations
            locX = [i for i in range(kernel_size,self.Y.shape[1]+1)]
            locY = [i for i in range(kernel_size,self.Y.shape[0]+1)]
            
            # Initialize output Y with original intensities then update by loop recursively
            Y_he = np.copy(self.Y)

            # Call standard HE algorithm for each window
            for y in tqdm(locY,desc='Local-HE'):
                for x in locX:
                    Y_window = self.Y[y-kernel_size:y,x-kernel_size:x] 
                    Y_he_window = self._apply(Y_window)[-1]
                    # Update Y-HE for corresponding window
                    Y_he[y-kernel_size:y,x-kernel_size:x] = Y_he_window

            # Original histogram calculation (for global)
            r_pmf = HE.getHistogram(self.Y.flatten(),levels=self.levels)[-1]
            tr = None

        elif mode == 2: # Histogram Specification 
            r_pmf,tr,Y_he = self._apply(self.Y,target_hist=target_hist)

        # Update variables
        self.Y_he = Y_he
        self.r_pmf = r_pmf # original-histogram
        self.s_pmf = HE.getHistogram(Y_he.flatten(),levels=self.levels)[-1] # he-image histogram 
        self.tr = tr # transform function

            
    def _apply(self,Y,target_hist=None):

        # Initialize output array with original image 1D*array
        Y_1d = Y.flatten()

        # Original-image histogram stats
        r = self.levels
        r_loc,_,r_pmf = HE.getHistogram(Y_1d,levels=r)

        # Determine transform function based on cumulative sum of probabilities
        tr = (self.L-1)*np.cumsum(r_pmf) # Transformation function
        s = np.clip(np.round(tr,0),0,self.L-1) # Round new intensity values to integer

        # [Histogram Specification] (Only for Mode 2)
        if target_hist is not None:
            z_pmf = target_hist 
            # Transformation function obtained from the specified histogram
            tr = (self.L-1)*np.cumsum(z_pmf) 
            z_g = np.clip(np.round(tr,0),0,self.L-1) # Round new intensity values to integer
            
            z = s # matching from s values to z corresponds
            for i in range(len(z)):
                z[i] = np.argmin(abs(z_g-z[i]))
            s = z 

        # Construct the HE-image by changing old intensities to equalized ones
        for i in r:
            Y_1d[r_loc[i]] = s[i]

        Y_he = Y_1d.reshape(Y.shape)

        return [r_pmf,tr,Y_he]  


    @staticmethod
    def getHistogram(array1d,levels):

        loc,hist,pmf = [],[],[]

        pixel_total = len(array1d)

        for i in levels:
            idx = np.where(array1d==i)[0]
            count = idx.shape[0]

            loc.append(idx)
            hist.append(count)
            pmf.append(count/pixel_total)

        return loc,hist,pmf

