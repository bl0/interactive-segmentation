"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import numpy as np
import os
import pydensecrf.densecrf as dcrf

from cv2 import imread, imwrite

from utils import load_image, load_mask, mask2color
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

if len(sys.argv) != 4:
    print("Usage: python {} IMAGE_DIR ANNO_DIR OUTPUT_DIR".format(sys.argv[0]))
    print("")
    print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
    sys.exit(1)


img_dir = sys.argv[1]
anno_dir = sys.argv[2]
output_dir = sys.argv[3]

for f in os.listdir(img_dir):
    if not ('.png' in f.lower() or '.jpg' in f.lower()):
        continue

    print('processing {}'.format(f))

    fn_im = os.path.join(img_dir, f)
    fn_anno = os.path.join(anno_dir, f)
    fn_output = os.path.join(output_dir, f)

    # Read images and annotation
    img = load_image(fn_im, 400, 600)
    mask = load_mask(fn_anno, img.shape)

    assert img.shape[:2] == mask.shape

    labels = mask.flatten()
    n_labels = 2

    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Do inference and compute MAP
    Q = d.inference(1)
    MAP = np.argmax(Q, axis=0)
    color = mask2color(MAP.reshape(img.shape[:2]))
    imwrite(fn_output, color)