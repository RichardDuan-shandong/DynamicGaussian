"""

    FileName          : const.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-08
    Description       : root para
    Version           : 1.0
    License           : MIT License
    
"""

MULTI_VIEW_DATA = 'nerf_data'
SEG_FOLDER = '/tmp_data/seg_data'
EPISODE_FOLDER = 'episode%d'
EPISODES_FOLDER = 'episodes'
VARIATIONS_ALL_FOLDER= 'all_variations'
POSE = 'poses'
IMAGES = 'images'
SAM2_CHECKPOINT = "../../third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "../third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CLIP_PATH = "../../openai/clip-vit-large-patch14"
DINO_PY_PATH = "../../third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHT_PATH = "../../third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SPACY_WEIGHT_PATH = "../../third_party/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0"
BOX_TRESHOLD = 0.18
TEXT_TRESHOLD = 0.2