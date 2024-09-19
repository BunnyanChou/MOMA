from typing import Tuple

import cv2
import numpy as np
import torch
import os

from src.config.default import get_cfg_defaults
from src.momamatcher import MOMAMatcher
from src.utils.misc import lower_config
from src.utils import viz2d
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_args():
    import argparse

    parser = argparse.ArgumentParser("test quadtree attention-based feature matching")
    parser.add_argument("--weight_path", default="./OUTPUT/densematching/AdaMatcher-640-bs2/version_19/checkpoints/epoch=2-auc@5=0.498-auc@10=0.666-auc@20=0.793.ckpt", 
                        type=str)
    parser.add_argument("--config_path", default="./configs/loftr/outdoor/loftr_ds_dense.py")
    parser.add_argument("--data_dir", default="./dataset/sunyatsen/3_auditorium/images/",
                        type=str)
    parser.add_argument("--output_dir", default="./exp_intermediate/",
                        type=str)
    parser.add_argument("--confidence_thresh", type=float, default=0.)

    return parser.parse_args()

def main():
    args = get_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    config = lower_config(config)

    matcher = MOMAMatcher(config=config["momamatcher"])
    state_dict = torch.load(args.weight_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict({k.replace('matcher.', ''): v for k, v in state_dict.items()})

    name0 = "X/RIEBO000164.JPG" #"phoenix/S6/zl548/MegaDepth_v1/0022/dense0/imgs/307036457_51029c5b2b_o.jpg" # #"A/RIEBO000671.JPG"#
    name1 = "D/RIEBO000412.JPG" #"phoenix/S6/zl548/MegaDepth_v1/0022/dense0/imgs/1247003203_b554be0557_o.jpg" # #"A/RIEBO001252.JPG"#
    
    query_image = cv2.imread(os.path.join(args.data_dir, name0), cv2.IMREAD_COLOR)
    ref_image = cv2.imread(os.path.join(args.data_dir, name1), cv2.IMREAD_COLOR)   

    
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
    new_shape = (480, 640)

    query_image_resize = cv2.resize(query_image, new_shape[::-1])
    ref_image_resize = cv2.resize(ref_image, new_shape[::-1])

    with torch.no_grad():
        batch = {
            "image0": load_torch_image(query_image_resize),
            "image1": load_torch_image(ref_image_resize),
        }

        matcher.eval()
        matcher.to("cuda")
        matcher(batch)

        query_kpts = batch["mkpts0_f"].cpu().numpy()
        ref_kpts = batch["mkpts1_f"].cpu().numpy()
        # confidences = batch["mconf"].cpu().numpy()
        del batch
        
        query_shape = query_image.shape[:2]
        ref_shape = ref_image.shape[:2]

        query_kpts = resample_kpts(
            query_kpts,
            float(query_shape[0]) / float(new_shape[0]),
            float(query_shape[1]) / float(new_shape[1]),
        )

        ref_kpts = resample_kpts(
            ref_kpts,
            float(ref_shape[0]) / float(new_shape[0]),
            float(ref_shape[1]) / float(new_shape[1]),
        )

        # conf_mask = np.where(confidences > args.confidence_thresh)
        # query_kpts = query_kpts[conf_mask]
        # ref_kpts = ref_kpts[conf_mask]
        num_matches = query_kpts.shape[0]
        
        axes = viz2d.plot_images([query_image, ref_image])
        viz2d.plot_matches(query_kpts, ref_kpts, color="lime", lw=0.2)
        viz2d.add_text(0, f'matches = {num_matches}', fs=20)
        viz2d.save_plot(os.path.join(args.output_dir, "-".join(["momamatcher_", name0.split("/")[-1], name1.split("/")[-1]])))
"""
    def _np_to_cv2_kpts(np_kpts):
        cv2_kpts = []
        for np_kpt in np_kpts:
            cur_cv2_kpt = cv2.KeyPoint()
            cur_cv2_kpt.pt = tuple(np_kpt)
            cv2_kpts.append(cur_cv2_kpt)
        return cv2_kpts

    query_shape = query_image.shape[:2]
    ref_shape = ref_image.shape[:2]
    query_kpts = resample_kpts(
        query_kpts,
        query_shape[0] / new_shape[0],
        query_shape[1] / new_shape[1],
    )

    
    query_kpts, ref_kpts = _np_to_cv2_kpts(query_kpts), _np_to_cv2_kpts(ref_kpts)

    matched_image = cv2.drawMatches(
        query_image,
        query_kpts,
        ref_image,
        ref_kpts,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(query_kpts))
        ],
        None,
        flags=2,
    )
    cv2.imwrite("result.jpg", matched_image)
"""

def resample_kpts(kpts: np.ndarray, height_ratio, width_ratio):
    kpts = kpts.astype(float)
    kpts[:, 0] *= float(width_ratio)
    kpts[:, 1] *= float(height_ratio)

    return kpts


def load_torch_image(image):
    image = torch.from_numpy(image.transpose(2, 0, 1))[None].cuda() / 255
    return image


if __name__ == "__main__":
    main()
