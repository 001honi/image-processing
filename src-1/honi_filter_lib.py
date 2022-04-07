"""
@Author: Honi, Selahaddin (honis@kaist.ac.kr)
@Date: 2022 March
"""
import cv2 as cv 
import numpy as np
from tqdm import tqdm

class ImgFilter():
    def __init__(self): 
        self.Y     = None # original-image intensity map 
        self.Cb    = None 
        self.Cr    = None
        self.Y_out = None # output-image intensity map
        

    def setInput(self,img_BGR):
        # Convert input image into YCrCb color-space and set Y intensity map
        img_YCrCb = cv.cvtColor(img_BGR,cv.COLOR_BGR2YCR_CB)
        (Y,Cr,Cb) = cv.split(img_YCrCb)

        self.Y = Y 
        self.Cb = Cb
        self.Cr = Cr


    def getOutput(self):
        img_YCrCb = cv.merge([self.Y_out,self.Cr,self.Cb])
        img_BGR = cv.cvtColor(img_YCrCb, cv.COLOR_YCR_CB2BGR)
        return img_BGR


    def applyGB(self,sigma=3):
        k = int(2*np.ceil(3*sigma)+1) #kernel size
        kernel = np.zeros((k,k))
        center = k//2

        # Construct Gaussian kernel
        for r in range(k):
            for c in range(k):
                i = abs(r-center); j = abs(c-center) # L1 distance to center of kernel
                kernel[r,c] = ImgFilter.gauss(sigma=sigma,dist=abs(i)+abs(j))

        # Slide kernel over the image
        Y_out = cv.filter2D(self.Y,ddepth=-1,kernel=kernel)
        self.Y_out = Y_out.astype('uint8')
       

    def applyBF(self,sigma_space=3,sigma_range=0.1,img_guidance=None):
        
        Y_out = self.Y.copy()
        Y_guidance = self.Y.copy()

        if img_guidance is not None: # Joint Bilateral Filter
            img_YCrCb = cv.cvtColor(img_guidance,cv.COLOR_BGR2YCR_CB)
            (Y_guidance,_,_) = cv.split(img_YCrCb)

        k = int(2*np.ceil(3*sigma_space)+1) #kernel size
        neighborhood = list(range(-k//2 +1,k//2 +1))
        H,W = Y_guidance.shape

        for y in tqdm(range(H)):
            for x in range(W):
            
                # check neighborhood inside the image
                if y < k//2 or y >= H-(k//2) \
                    or x < k//2 or x >= W-(k//2):
                    continue

                # y,x center point location
                # i,j relative neighborhood distance to center point 

                Ic = self.Y[y,x] # intensity of the center point 
                Ic_out = 0 # new intensity of the center point (will update by loop)
                C = 0 # normalization constant (will update by loop)

                for j in neighborhood:
                    for i in neighborhood:

                        In = Y_guidance[y-j,x-i] # neighbor pixel intensity 

                        Kd = ImgFilter.gauss(sigma=sigma_space,dist=abs(i)+abs(j))
                        Kr = ImgFilter.gauss(sigma=sigma_range,dist=abs(Ic-In))

                        Ic_out += Kd*Kr*In
                        C += Kd*Kr

                # Update the center pixel intensity after normalized by C
                Y_out[y,x] = Ic_out/C 
        
        self.Y_out = Y_out

    
    def applyRollingGuidance(self,sigma=3,sigma_space=3,sigma_range=0.2,iteration=5,imwrite=True):

        self.applyGB(sigma)
        img_guidance_step1 = self.getOutput()

        img_guidance = img_guidance_step1

        for i in range(iteration):

            if imwrite:
                cv.imwrite(f'out-step{i+1}.png',img_guidance)

            self.applyBF(sigma_space,sigma_range,img_guidance)
            img_guidance = self.getOutput()

        self.Y_out = cv.split(img_guidance)[0]


    @staticmethod
    def gauss(sigma,dist):
        return (1/(2*np.pi*sigma**2))*np.exp(-(dist**2)/(2*sigma**2))  