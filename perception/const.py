"""

    FileName          : const.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-08
    Description       : root para
    Version           : 1.0
    License           : MIT License
    
"""
import numpy as np

MULTI_VIEW_DATA = 'nerf_data'
SEG_FOLDER = '/tmp_data/seg_data'
EPISODE_FOLDER = 'episode%d'
EPISODES_FOLDER = 'episodes'
VARIATIONS_ALL_FOLDER= 'all_variations'
POSE = 'poses'
IMAGES = 'images'
DEPTHS = 'depths'
SAM2_CHECKPOINT = "../../third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "../../third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CLIP_PATH = "../../openai/clip-vit-large-patch14"
DINO_PY_PATH = "../../third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHT_PATH = "../../third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SPACY_WEIGHT_PATH = "../../third_party/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0"
BOX_TRESHOLD = 0.18
TEXT_TRESHOLD = 0.2
DEFAULT_RGB_SCALE_FACTOR = 2**24 - 1
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                            np.uint16: 1000.0,
                            np.int32: DEFAULT_RGB_SCALE_FACTOR}
ZFAR = 10
ZNEAR = 10e-2