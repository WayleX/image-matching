import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
from algorithm import get_matching
import torch

def draw(path_img1:str, path_img2:str):
    #just load images and get matchings from algorithm
    mkpts0,mkpts1,inliers = get_matching(path_img1, path_img2)
    img1 = K.io.load_image(path_img1, K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(path_img2, K.io.ImageLoadType.RGB32)[None, ...]
    img1 = K.geometry.resize(img1, (375, 500), antialias=True)
    img2 = K.geometry.resize(img2, (375, 500), antialias=True)
    #Draw 
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts0).view(1, -1, 2),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1, -1, 2),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        ),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={"inlier_color": (0, 1, 0), "tentative_color": (1,0,0), "feature_color": None, "vertical": False},
    )