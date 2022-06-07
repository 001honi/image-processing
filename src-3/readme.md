<br />
<p align="center">
  <h1 align="center">Image Inpainting</h2>

  <p align="center"><a href="./report.pdf"><strong>Follow the report »</strong></a>
  </p>
</p>

@Source reference:
https://github.com/igorcmoura/inpaint-object-remover

@About:
Reference code is modified to implement original idea.

[1] Modified methods in `inpainter.py` script are specified by highlighted comments.  
  
	The most significant change is in line 197
	_find_source_patch()

[2] PATCH_NUM and LAMBDA_DIST args are added for usage:  

usage: inpainter [-h] [-ps PATCH_SIZE] [-pn PATCH_NUM] [-dist LAMBDA_DIST]  
                 [-o OUTPUT] [--plot-progress] [--input_image INPUT_IMAGE]  
                 [--mask MASK]  
  
optional arguments:  
  -h, --help  
  show this help message and exit  
  -ps PATCH_SIZE, --patch-size PATCH_SIZE  
                        the size of the patches  
  -pn PATCH_NUM, --patch-num PATCH_NUM  
                        the number of patches combined in source region  
  -dist LAMBDA_DIST, --lambda-dist LAMBDA_DIST  
                        regularization for euclidean distance  
  -o OUTPUT, --output OUTPUT  
                        the file path to save the output image  
  --plot-progress  
  plot each generated image  
  --input_image INPUT_IMAGE  
                        the image containing objects to be removed  
  --mask MASK  
  the mask of the region to be removed  







