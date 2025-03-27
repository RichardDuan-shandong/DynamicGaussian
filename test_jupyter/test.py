import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.build_sam import build_sam2
import matplotlib.patches as patches
import cv2
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from openai.simple_tokenizer import SimpleTokenizer
import regex as re
import ftfy
from torchvision.ops import box_convert
import html
import pycocotools.mask as coco_mask
# 设备选择
device = torch.device("cuda")

# 加载SAM模型(model_cfg从build_sam2所在开始计算)
sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# 均匀采样SAM
rough_mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=19,
    points_per_batch=10,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.75,
    stability_score_offset=1,
    crop_n_layers=0,
    box_nms_thresh=0.85,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=120,
    use_m2m=False,
)

# 点指定采样SAM
specific_mask_predictor = SAM2ImagePredictor(
    sam_model=sam2,
    mask_threshold=0.5,
    max_hole_area=0.0,
    max_sprinkle_area=0.0,
)

# 加载clip模型
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # clip模型
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14") # 生成器
# 加载dino(路径是从该程序开始计算)
dino = load_model("third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth")

# 读取image
images = []
images_input = []
images_dino_input = []
for i in range(0, 20):
    image = Image.open(f'test_jupyter/images/{i}.png')
    image_input = np.array(image.convert("RGB"))
    images.append(image)
    images_input.append(image_input)
    image_source, image_dino_input = load_image(f'test_jupyter/images/{i}.png')
    images_dino_input.append(image_dino_input)

# 给出场景任务描述
task_description = "A RobotArm close_jar"

# 处理lang_prompt
def basic_clean(text):
    text = ftfy.fix_text(text)  # 修复文本编码问题
    text = html.unescape(html.unescape(text))  # 处理 HTML 转义字符
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)  # 将连续空白字符替换为单个空格
    text = text.strip()  # 去除开头和结尾的空白字符
    return text

def remove_underscores(text):
    text = re.sub(r'_', ' ', text)  # 将下划线替换为空格
    return text

def preprocess_text(text):
    text = basic_clean(text)
    text = whitespace_clean(text)
    text = remove_underscores(text)
    text = text.lower()  # 转换为小写
    pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
    tokens = re.findall(pat, text)
    return tokens

# 预处理文本获得去掉特殊符号后的分词模型(去除下划线和空格等，并统一为小写)
cleaned_text = preprocess_text(task_description)
task_description = " ".join(cleaned_text)

'''
    获得任务的全局文本描述embedding(a robot arm close jar -> torch.Size([1, 768])) : lang_embedding
'''
inputs = processor(text=task_description , return_tensors="pt")
lang_embedding = clip.get_text_features(**inputs)
# print(lang_embedding.shape)

'''
    获得场景操作(包含一些物体以及action)的token (['robotarm', 'close', 'jar'] -> torch.Size([3, 768])) : mani_tokens
'''
with torch.no_grad():
    mani_tokens = torch.empty((0, 768))
    stopwords = {"a", "the", "and", "an", "of", "in", "on", "at", "to", "with", "for", "by"}    # 去除task_description中的连接词与介词
    mani_words = [word for word in cleaned_text if word.lower() not in stopwords]

    for mani_word in mani_words:
        inputs = processor(text=mani_word , return_tensors="pt")
        mani_tokens = torch.cat((mani_tokens, clip.get_text_features(**inputs)), dim=0)
    # print(mani_tokens.shape)

TEXT_PROMPT = " . ".join(mani_words) + "."
BOX_TRESHOLD = 0.1
TEXT_TRESHOLD = 0.1

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))  # 确保有 Figure 和 Axes
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)  # 这里仍然是叠加
    plt.axis("off")
    plt.show()



def show_masks_on_image(image, masks, alpha=0.5, show_borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    mask_overlay = np.zeros_like(image, dtype=np.float32)

    for i, mask in enumerate(masks):
        color = np.random.rand(3)  # 随机颜色
        mask_overlay[mask > 0] = color

        if show_borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(mask_overlay, [contour], -1, (0, 0, 1), thickness=1)

    plt.imshow(mask_overlay, alpha=alpha)
    plt.axis("off")
    plt.title("Segmented Masks on Image")
    plt.show()
    
    
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pycocotools.mask as coco_mask

def extract_masked_objects(image, anns, padding=3):
    """
    从 image 提取 mask 对应的目标部分，并去除背景
    :param image: 原图 (H, W, 3)
    :param anns: mask annotations (list of dict)
    :param padding: 裁剪时额外添加的像素
    :return: 裁剪后的目标部分 (list of PIL Image)
    """
    height, width = image.shape[:2]
    extracted_objects = []

    for ann in anns:
        mask = ann.get("segmentation", None)  # segmentation 是布尔数组 (H, W)

        if mask is None or mask.size == 0:
            print("Warning: Empty segmentation found!")
            continue

        # 计算 bounding box
        y_indices, x_indices = np.where(mask)  # 获取 True（被遮盖区域）的坐标
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue  # 避免空 mask

        x1, x2 = max(x_indices.min() - padding, 0), min(x_indices.max() + padding, width)
        y1, y2 = max(y_indices.min() - padding, 0), min(y_indices.max() + padding, height)

        # 生成带透明度的图像
        masked_img = np.zeros((height, width, 4), dtype=np.uint8)
        masked_img[:, :, :3] = image  # 复制 RGB 部分
        masked_img[:, :, 3] = (mask * 255).astype(np.uint8)  # 设置 Alpha 通道：被遮盖区域 255，其他 0

        # 裁剪目标区域
        cropped_img = masked_img[y1:y2, x1:x2]

        # 转换为 PIL Image 方便可视化
        extracted_objects.append(Image.fromarray(cropped_img, mode="RGBA"))

    return extracted_objects

for i in range(0, 20):
    image, image_input, image_dino_input = images[i], images_input[i], images_dino_input[i]
    # rough_masks = rough_mask_generator.generate(image_input)
    # # 先去对获得的图像进行预处理
    # masked_images = extract_masked_objects(image_input, rough_masks)
    # mask_indices= 0
    # mani_masks = torch.empty(1, 768)
    # for obj in masked_images:
    #     # 将图片嵌入
    #     inputs_image = processor(images=image, return_tensors="pt")
    #     image_features = clip.get_image_features(**inputs_image)
    #     mani_masks = torch.cat((mani_masks, image_features), dim = 0)
    #     plt.figure()
    #     plt.imshow(obj)
    #     plt.axis("off")
    #     plt.show()
    #     mask_indices = mask_indices + 1 # 每次进行计数

    TEXT_PROMPT = "robot arm . jar . bottle cap."
    boxes, logits, phrases = predict(
        model=dino,
        image=image_dino_input,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    h, w, _ = image.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    rough_masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_xyxy,
        multimask_output=False,
    )
    show_masks_on_image(image, rough_masks)
    image_source, image = load_image(IMAGE_PATH)

