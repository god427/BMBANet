# import os
# import os.path as osp
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2
# import matplotlib.pyplot as plt
# import albumentations as albu
# from .transform import *
# import matplotlib.patches as mpatches
# from PIL import Image
# import random
#
# # WHU-OPT-SAR数据集类别定义
# CLASSES = ('background', 'farmland', 'city', 'village', 'water', 'forest', 'road', 'others')
# PALETTE = [
#     [0, 0, 0],  # background - 黑色
#     [0, 102, 204],  # farmland - 棕色
#     [255, 0, 0],  # city - 红色
#     [255, 255, 0],  # village - 黄色
#     [0, 0, 255],  # water - 蓝色
#     [0, 167, 85],  # forest - 绿色
#     [0, 255, 255],  # road - 青色
#     [153, 102, 153]  # others - 紫色
# ]
#
# # 图像尺寸保持不变
# ORIGIN_IMG_SIZE = (1024, 1024)
# INPUT_IMG_SIZE = (1024, 1024)
# TEST_IMG_SIZE = (1024, 1024)
#
#
# def get_training_transform():
#     train_transform = [
#         # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
#         # albu.RandomRotate90(p=0.25),
#         albu.Normalize()
#     ]
#     return albu.Compose(train_transform)
#
#
# def train_aug(img, mask):
#     crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
#                         SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
#     img, mask = crop_aug(img, mask)
#     img, mask = np.array(img), np.array(mask)
#     aug = get_training_transform()(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask
#
#
# def get_val_transform():
#     val_transform = [
#         albu.Normalize()
#     ]
#     return albu.Compose(val_transform)
#
#
# def val_aug(img, mask):
#     img, mask = np.array(img), np.array(mask)
#     aug = get_val_transform()(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask
#
#
# class WHUOptSarDataset(Dataset):
#     def __init__(self, data_root='/mnt/e/sjj/WHU/3/test', mode='val', img_dir='images', mask_dir='labels',
#                  img_suffix='.tif', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
#                  img_size=ORIGIN_IMG_SIZE):
#         self.data_root = data_root
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_suffix = img_suffix
#         self.mask_suffix = mask_suffix
#         self.transform = transform
#         self.mode = mode
#         self.mosaic_ratio = mosaic_ratio
#         self.img_size = img_size
#         self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
#
#     def __getitem__(self, index):
#         p_ratio = random.random()
#         if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
#             img, mask = self.load_img_and_mask(index)
#             if self.transform:
#                 img, mask = self.transform(img, mask)
#         else:
#             img, mask = self.load_mosaic_img_and_mask(index)
#             if self.transform:
#                 img, mask = self.transform(img, mask)
#
#         img = torch.from_numpy(img).permute(2, 0, 1).float()
#         mask = torch.from_numpy(mask).long()
#         img_id = self.img_ids[index]
#         results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
#         return results
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def get_img_ids(self, data_root, img_dir, mask_dir):
#         img_filename_list = os.listdir(osp.join(data_root, img_dir))
#         mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
#         assert len(img_filename_list) == len(mask_filename_list)
#         img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
#         return img_ids
#
#     def load_img_and_mask(self, index):
#         img_id = self.img_ids[index]
#         img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
#         mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
#
#         # 加载光学图像
#         img = Image.open(img_name).convert('RGB')
#
#         # 加载标签图像并转换为类别索引
#         mask = Image.open(mask_name)
#         if mask.mode == 'RGB':
#             # 如果标签是RGB图像，需要根据调色板转换为类别索引
#             mask = self.rgb_mask_to_index(mask)
#         else:
#             # 如果已经是索引图像，直接使用
#             mask = mask.convert('L')
#
#         return img, mask
#
#     def rgb_mask_to_index(self, rgb_mask):
#         """将RGB掩码图像转换为类别索引图像"""
#         rgb_array = np.array(rgb_mask)
#         h, w, c = rgb_array.shape
#         index_mask = np.zeros((h, w), dtype=np.uint8)
#
#         # 为每个类别创建掩码
#         for class_idx, color in enumerate(PALETTE):
#             # 找到与当前颜色匹配的像素
#             color_match = np.all(rgb_array == color, axis=2)
#             index_mask[color_match] = class_idx
#
#         return Image.fromarray(index_mask, mode='L')
#
#     def load_mosaic_img_and_mask(self, index):
#         indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
#         img_a, mask_a = self.load_img_and_mask(indexes[0])
#         img_b, mask_b = self.load_img_and_mask(indexes[1])
#         img_c, mask_c = self.load_img_and_mask(indexes[2])
#         img_d, mask_d = self.load_img_and_mask(indexes[3])
#
#         img_a, mask_a = np.array(img_a), np.array(mask_a)
#         img_b, mask_b = np.array(img_b), np.array(mask_b)
#         img_c, mask_c = np.array(img_c), np.array(mask_c)
#         img_d, mask_d = np.array(img_d), np.array(mask_d)
#
#         w = self.img_size[1]
#         h = self.img_size[0]
#
#         start_x = w // 4
#         strat_y = h // 4
#         # The coordinates of the splice center
#         offset_x = random.randint(start_x, (w - start_x))
#         offset_y = random.randint(strat_y, (h - strat_y))
#
#         crop_size_a = (offset_x, offset_y)
#         crop_size_b = (w - offset_x, offset_y)
#         crop_size_c = (offset_x, h - offset_y)
#         crop_size_d = (w - offset_x, h - offset_y)
#
#         random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
#         random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
#         random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
#         random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])
#
#         croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
#         croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
#         croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
#         croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())
#
#         img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
#         img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
#         img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
#         img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']
#
#         top = np.concatenate((img_crop_a, img_crop_b), axis=1)
#         bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
#         img = np.concatenate((top, bottom), axis=0)
#
#         top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
#         bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
#         mask = np.concatenate((top_mask, bottom_mask), axis=0)
#         mask = np.ascontiguousarray(mask)
#         img = np.ascontiguousarray(img)
#
#         img = Image.fromarray(img)
#         mask = Image.fromarray(mask)
#
#         return img, mask


import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
# 注意：若transform.py不在当前路径，需确认导入路径正确性
from .transform import *
import matplotlib.patches as mpatches
from PIL import Image
import random

# WHU-OPT-SAR数据集类别定义
CLASSES = ('background', 'farmland', 'city', 'village', 'water', 'forest', 'road', 'others')
PALETTE = [
    [0, 0, 0],  # background - 黑色
    [204, 102, 0],  # farmland - 棕色
    [255, 0, 0],  # city - 红色
    [255, 255, 0],  # village - 黄色
    [0, 0, 255],  # water - 蓝色
    [85, 167, 0],  # forest - 绿色
    [0, 255, 255],  # road - 青色
    [153, 102, 153]  # others - 紫色
]

# 图像尺寸保持不变吧  
ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)

# --------------------- 新增：模型输入波段数配置 ---------------------
# 根据你的模型实际输入波段数修改（WHU-OPT-SAR光学图像默认RGB=3波段）
MODEL_INPUT_BANDS = 4
# 标记是否已打印（避免重复打印）
is_printed = False


def get_training_transform():
    train_transform = [
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        # albu.RandomRotate90(p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class WHUOptSarDataset(Dataset):
    def __init__(self, data_root=r'E:\sjj\WHU2\WHU\test', mode='val', img_dir='images', mask_dir='labels',
                 img_suffix='.tif', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 # img_size=ORIGIN_IMG_SIZE, model_input_bands=MODEL_INPUT_BANDS):
                 img_size=ORIGIN_IMG_SIZE, model_input_bands=3):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
        # --------------------- 新增：保存模型输入波段数 ---------------------
        self.model_input_bands = model_input_bands

    def __getitem__(self, index):
        global is_printed
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            # --------------------- 新增：打印图像原始波段数（首次加载时） ---------------------
            if not is_printed:
                # 获取图像原始波段数
                if img.mode == 'RGB':
                    img_original_bands = 3
                elif img.mode == 'RGBA':
                    img_original_bands = 4
                elif img.mode == 'L':
                    img_original_bands = 1
                else:
                    img_original_bands = len(img.getbands())  # 通用获取波段数方式
                # 打印核心信息
                print(f"=== WHU-OPT-SAR数据集波段信息 ===")
                print(f"图像原始波段数：{img_original_bands}")
                print(f"模型使用的波段数：{self.model_input_bands}")
                is_printed = True  # 标记已打印，避免重复输出
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            # --------------------- 新增：mosaic模式下也打印（首次加载时） ---------------------
            if not is_printed:
                if img.mode == 'RGB':
                    img_original_bands = 3
                elif img.mode == 'RGBA':
                    img_original_bands = 4
                elif img.mode == 'L':
                    img_original_bands = 1
                else:
                    img_original_bands = len(img.getbands())
                print(f"=== WHU-OPT-SAR数据集波段信息（Mosaic模式） ===")
                print(f"图像原始波段数：{img_original_bands}")
                print(f"模型使用的波段数：{self.model_input_bands}")
                is_printed = True
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        # 加载光学图像
        img = Image.open(img_name).convert('RGBA')

        # 加载标签图像并转换为类别索引
        mask = Image.open(mask_name)
        if mask.mode == 'RGBA':
            # 如果标签是RGB图像，需要根据调色板转换为类别索引
            mask = self.rgb_mask_to_index(mask)
        else:
            # 如果已经是索引图像，直接使用
            mask = mask.convert('L')

        return img, mask

    def rgb_mask_to_index(self, rgb_mask):
        """将RGB掩码图像转换为类别索引图像"""
        rgb_array = np.array(rgb_mask)
        h, w, c = rgb_array.shape
        index_mask = np.zeros((h, w), dtype=np.uint8)

        # 为每个类别创建掩码
        for class_idx, color in enumerate(PALETTE):
            # 找到与当前颜色匹配的像素
            color_match = np.all(rgb_array == color, axis=2)
            index_mask[color_match] = class_idx

        return Image.fromarray(index_mask, mode='L')

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask