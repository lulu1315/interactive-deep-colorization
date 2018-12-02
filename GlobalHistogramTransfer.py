import sys, getopt

if len(sys.argv) != 8:
    print ('usage : python GlobalHistogramTransfer.py imatocolorize refimage outputimage prototxt caffemodel globalprototxt globalcaffemodel')
    sys.exit()
    
import cv2
import caffe
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from data import colorize_image as CI
from skimage import color
from data import lab_gamut as lab

#image to colorize
#img_path = '/mnt/Nanterre/Lulu/Projets/Colorisation/TheKid/waifu/waifu.1729.png'
img_path = sys.argv[1]
# color histogram to use
#ref_path = '/mnt/Nanterre/Lulu/Projets/RuPaul/style/finger1.jpg'
ref_path = sys.argv[2]
out_path = sys.argv[3]
prototxt = sys.argv[4]
caffemodel = sys.argv[5]
globprototxt = sys.argv[6]
globcaffemodel = sys.argv[7]

#print img_path
#print ref_path
#print out_path
#print prototxt
#print caffemodel
#sys.exit()

gpu_id = 0 # gpu to use
Xd = 256

# Colorization network
cid = CI.ColorizeImageCaffeGlobDist(Xd)

#cid.prep_net(gpu_id,prototxt_path='/mnt/shared/v16/ideepcolor/models/global_model/deploy_nodist.prototxt',caffemodel_path='/mnt/shared/v16/ideepcolor/models/global_model/global_model.caffemodel')
cid.prep_net(gpu_id,prototxt_path=prototxt,caffemodel_path=caffemodel)

# Global distribution network - extracts global color statistics from an image
#gt_glob_net = caffe.Net('/mnt/shared/v16/ideepcolor/models/global_model/global_stats.prototxt','/mnt/shared/v16/ideepcolor/models/global_model/dummy.caffemodel', caffe.TEST)
gt_glob_net = caffe.Net(globprototxt,globcaffemodel, caffe.TEST)

## Colorization in automatic mode (with no reference histogram)"
# Load image
cid.load_image(img_path)

# Dummy variables
input_ab = np.zeros((2,Xd,Xd))
input_mask = np.zeros((1,Xd,Xd))

# Colorization without global histogram
#img_pred = cid.net_forward(input_ab,input_mask);
#img_pred_auto_fullres = cid.get_img_fullres()

# Colorization with reference global histogram
def get_global_histogram(img_path):
    ref_img_fullres = caffe.io.load_image(ref_path)
    img_glob_dist = (255*caffe.io.resize_image(ref_img_fullres,(Xd,Xd))).astype('uint8') # load image
    gt_glob_net.blobs['img_bgr'].data[...] = img_glob_dist[:,:,::-1].transpose((2,0,1)) # put into 
    gt_glob_net.forward();
    glob_dist_in = gt_glob_net.blobs['gt_glob_ab_313_drop'].data[0,:-1,0,0].copy()
    return (glob_dist_in,ref_img_fullres)

(glob_dist_ref,ref_img_fullres) = get_global_histogram(ref_path)
img_pred = cid.net_forward(input_ab,input_mask,glob_dist_ref);
img_pred_withref_fullres = cid.get_img_fullres()

cv2.imwrite(out_path,cv2.cvtColor(img_pred_withref_fullres,cv2.COLOR_BGR2RGB))
#plt.imshow(img_pred_withref_fullres, interpolation='nearest')
#plt.show()
