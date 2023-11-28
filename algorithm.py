import cv2
import kornia as K
import kornia.feature as KF
import torch

def get_matching(path_to_photo_1 : str, path_to_photo_2 : str):
    #load and resize images with Kornia
    img1 = K.io.load_image(path_to_photo_1, K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(path_to_photo_2, K.io.ImageLoadType.RGB32)[None, ...]
    img1 = K.geometry.resize(img1, (375, 500), antialias=True)
    img2 = K.geometry.resize(img2, (375, 500), antialias=True)

    #Using LoFTR
    matcher = KF.LoFTR(pretrained="outdoor")

    #As LoFTR works only on grayscale images, we need to convert
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    #Use MAGSAC
    _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0
    return mkpts0, mkpts1, inliers