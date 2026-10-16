# import os
# import os.path as osp
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2
# import matplotlib.pyplot as plt
# import albumentations as albu
# from .transform import *   # 请确保此模块包含 Compose, RandomScale, SmartCropV1
# import matplotlib.patches as mpatches
# from PIL import Image
# import random
# import rasterio
#
# # -------------------- 修改1：尺寸常量设为 256 --------------------
# ORIGIN_IMG_SIZE = (256, 256)
# INPUT_IMG_SIZE = (256, 256)
# TEST_IMG_SIZE = (256, 256)
# # ----------------------------------------------------------------
#
# # -------------------- 自定义类别和颜色映射 --------------------
# CLASSES = ('background','water', 'vegetation', 'road', 'bareland', 'buildings')
# PALETTE = [
#     [0, 0, 0],
#     [0, 0, 255],       # 水体 - 蓝色
#     [0, 128, 0],       # 植被 - 绿色
#     [0, 255, 255],     # 道路 - 青色
#     [153, 99, 0],      # 裸地 - 褐色
#     [255, 0, 0]        # city - 红色
# ]
# IGNORE_INDEX = len(CLASSES)   # 忽略索引设为 5
# # ----------------------------------------------------------------
#
# def get_training_transform():
#     mean = [0.485, 0.456, 0.406, 0.5]   # RGB + NIR 示例值
#     std  = [0.229, 0.224, 0.225, 0.2]
#     train_transform = [
#         albu.RandomRotate90(p=0.25),
#         albu.HorizontalFlip(p=0.5),
#         albu.VerticalFlip(p=0.5),
#         albu.Normalize(mean=mean, std=std)
#     ]
#     return albu.Compose(train_transform)
#
# # def train_aug(img, mask):
# #     # -------------------- 修改2：裁剪尺寸改为 256 --------------------
# #     crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
# #                         SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=IGNORE_INDEX, nopad=False)])
# #     # ----------------------------------------------------------------
# #     img, mask = crop_aug(img, mask)
# #     img, mask = np.array(img), np.array(mask)
# #     aug = get_training_transform()(image=img.copy(), mask=mask.copy())
# #     img, mask = aug['image'], aug['mask']
# #     return img, mask
# def train_aug(img, mask):
#     from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1
#     crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
#                         SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=IGNORE_INDEX, nopad=False)])
#     img, mask = crop_aug(img, mask)
#     img, mask = np.array(img), np.array(mask)
#     # 使用自定义的4通道归一化（保留几何增强）
#     transform = albu.Compose([
#         albu.RandomRotate90(p=0.25),
#         albu.HorizontalFlip(p=0.5),
#         albu.VerticalFlip(p=0.5),
#         albu.Normalize(mean=[0.485,0.456,0.406,0.5], std=[0.229,0.224,0.225,0.2])
#     ])
#     aug = transform(image=img, mask=mask)
#     return aug['image'], aug['mask']
#
# def get_val_transform():
#     # val_transform = [
#     #     albu.Normalize()
#     # ]
#     # return albu.Compose(val_transform)
#     mean = [0.485, 0.456, 0.406, 0.5]
#     std  = [0.229, 0.224, 0.225, 0.2]
#     val_transform = [albu.Normalize(mean=mean, std=std)]
#     return albu.Compose(val_transform)
#
# # def val_aug(img, mask):
# #     img, mask = np.array(img), np.array(mask)
# #     aug = get_val_transform()(image=img.copy(), mask=mask.copy())
# #     img, mask = aug['image'], aug['mask']
# #     return img, mask
# def val_aug(img, mask):
#     img, mask = np.array(img), np.array(mask)
#     transform = albu.Compose([
#         albu.Resize(height=256, width=256),
#         albu.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
#     ])
#     augmented = transform(image=img, mask=mask)
#     return augmented['image'], augmented['mask']
#
# def get_test_transform():
#     test_transform = [
#         albu.Normalize()
#     ]
#     return albu.Compose(test_transform)
#
# def test_aug(img, mask):
#     img, mask = np.array(img), np.array(mask)
#     aug = get_test_transform()(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask
#
# class LYY(Dataset):
#     def __init__(self, data_root=r'E:/sjj/LYY/val', mode='val', img_dir='images', mask_dir='labels',
#                  img_suffix='.tif', mask_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
#                  img_size=ORIGIN_IMG_SIZE):
#         self.data_root = data_root
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_suffix = img_suffix
#         self.mask_suffix = mask_suffix
#         self.transform = transform
#         self.mode = mode
#         self.mosaic_ratio = mosaic_ratio
#         self.img_size = img_size    # 此时 img_size 为 (256, 256)
#         self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
#         # 构建颜色到索引的映射字典
#         self.color2index = self._build_color2index()
#
#     def _build_color2index(self):
#         color2index = {}
#         for idx, color in enumerate(PALETTE):
#             color2index[tuple(color)] = idx
#         return color2index
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
#     # def load_img_and_mask(self, index):
#     #     img_id = self.img_ids[index]
#     #     img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
#     #     mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
#     #
#     #     img = Image.open(img_name).convert('RGBA')
#     #     mask_rgb = np.array(Image.open(mask_name).convert('RGB'))
#     #
#     #     h, w = mask_rgb.shape[:2]
#     #     mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
#     #     for color, idx in self.color2index.items():
#     #         mask[(mask_rgb == color).all(axis=2)] = idx
#     #
#     #     mask = Image.fromarray(mask, mode='L')
#     #     return img, mask
#     def load_img_and_mask(self, index):
#         img_id = self.img_ids[index]
#         img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
#         mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
#
#         # ----- 使用 rasterio 读取多波段图像 -----
#         with rasterio.open(img_name) as src:
#             img = src.read()  # shape: (bands, H, W)
#         # 转置为 (H, W, bands)
#         img = np.transpose(img, (1, 2, 0))
#
#         # 数据类型转换（假设原始为 uint16 或 float，需要映射到 0-255）
#         # 方案1：如果原始数据已经是 uint8，直接转换
#         if img.dtype == np.uint16:
#             # 例如 16-bit 到 8-bit：除以 256（或根据实际范围线性拉伸）
#             img = (img / 256).astype(np.uint8)
#         elif img.dtype == np.float32:
#             # 假设范围 0-1 映射到 0-255
#             img = (img * 255).astype(np.uint8)
#         else:
#             img = img.astype(np.uint8)
#
#         # ----- 读取掩码（保持原样，RGB 彩色标签）-----
#         mask_rgb = np.array(Image.open(mask_name).convert('RGB'))
#         h, w = mask_rgb.shape[:2]
#         mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
#         for color, idx in self.color2index.items():
#             mask[(mask_rgb == color).all(axis=2)] = idx
#
#         # 返回 PIL Image 对象以保持后续接口一致（也可返回 numpy，需调整 train_aug）
#         # 注意：PIL 对于多波段图像有限制（最多 4 通道），若波段数 >4 则不能转为 PIL
#         # 建议修改 train_aug/val_aug 直接处理 numpy 数组，跳过 PIL 转换。
#         # 但为了最小改动，可以继续转 PIL（仅支持 3 或 4 通道）。
#         if img.shape[2] <= 4:
#             img = Image.fromarray(img)  # 如果是 4 通道，模式为 'RGBA'；3 通道为 'RGB'
#         else:
#             # 波段数超过 4，不能转 PIL，需要修改 train_aug 接受 numpy 数组
#             raise NotImplementedError("超过4通道的图像不能转为 PIL，请修改增强函数直接处理 numpy 数组。")
#         mask = Image.fromarray(mask, mode='L')
#         return img, mask
#
#     def load_mosaic_img_and_mask(self, index):
#         # 马赛克增强，最终图像尺寸为 self.img_size = (256, 256)
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
#         return img, mask
#
#
# # import os
# # import os.path as osp
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # import cv2
# # import matplotlib.pyplot as plt
# # import albumentations as albu
# # from .transform import *   # 请确保此模块包含 Compose, RandomScale, SmartCropV1
# # import matplotlib.patches as mpatches
# # from PIL import Image
# # import random
# # # import rasterio          # 新增：用于读取多通道图像
# #
# # # -------------------- 修改1：尺寸常量设为 256 --------------------
# # ORIGIN_IMG_SIZE = (256, 256)
# # INPUT_IMG_SIZE = (256, 256)
# # TEST_IMG_SIZE = (256, 256)
# # # ----------------------------------------------------------------
# #
# # # -------------------- 自定义类别和颜色映射 --------------------
# # CLASSES = ('background','water', 'vegetation', 'road', 'bareland', 'buildings')
# # PALETTE = [
# #     [0, 0, 0],
# #     [0, 0, 255],       # 水体 - 蓝色
# #     [0, 128, 0],       # 植被 - 绿色
# #     [0, 255, 255],     # 道路 - 青色
# #     [153, 99, 0],      # 裸地 - 褐色
# #     [255, 0, 0]        # city - 红色
# # ]
# # IGNORE_INDEX = len(CLASSES)   # 忽略索引设为 5
# # # ----------------------------------------------------------------
# #
# # def get_training_transform():
# #     train_transform = [
# #         albu.RandomRotate90(p=0.25),
# #         albu.HorizontalFlip(p=0.5),
# #         albu.VerticalFlip(p=0.5),
# #         albu.Normalize()
# #     ]
# #     return albu.Compose(train_transform)
# #
# # def train_aug(img, mask):
# #     # -------------------- 修改2：裁剪尺寸改为 256 --------------------
# #     crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
# #                         SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=IGNORE_INDEX, nopad=False)])
# #     # ----------------------------------------------------------------
# #     img, mask = crop_aug(img, mask)
# #     img, mask = np.array(img), np.array(mask)
# #     aug = get_training_transform()(image=img.copy(), mask=mask.copy())
# #     img, mask = aug['image'], aug['mask']
# #     return img, mask
# #
# # def get_val_transform():
# #     val_transform = [
# #         albu.Normalize()
# #     ]
# #     return albu.Compose(val_transform)
# #
# # def val_aug(img, mask):
# #     img, mask = np.array(img), np.array(mask)
# #     aug = get_val_transform()(image=img.copy(), mask=mask.copy())
# #     img, mask = aug['image'], aug['mask']
# #     return img, mask
# #
# # def get_test_transform():
# #     test_transform = [
# #         albu.Normalize()
# #     ]
# #     return albu.Compose(test_transform)
# #
# # def test_aug(img, mask):
# #     img, mask = np.array(img), np.array(mask)
# #     aug = get_test_transform()(image=img.copy(), mask=mask.copy())
# #     img, mask = aug['image'], aug['mask']
# #     return img, mask
# #
# # class LYY(Dataset):
# #     def __init__(self, data_root=r'/mnt/e/sjj/LYY/val', mode='val', img_dir='images', mask_dir='labels',
# #                  img_suffix='.tif', mask_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
# #                  img_size=ORIGIN_IMG_SIZE):
# #         self.data_root = data_root
# #         self.img_dir = img_dir
# #         self.mask_dir = mask_dir
# #         self.img_suffix = img_suffix
# #         self.mask_suffix = mask_suffix
# #         self.transform = transform
# #         self.mode = mode
# #         self.mosaic_ratio = mosaic_ratio
# #         self.img_size = img_size    # 此时 img_size 为 (256, 256)
# #         self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
# #         # 构建颜色到索引的映射字典
# #         self.color2index = self._build_color2index()
# #
# #     def _build_color2index(self):
# #         color2index = {}
# #         for idx, color in enumerate(PALETTE):
# #             color2index[tuple(color)] = idx
# #         return color2index
# #
# #     def __getitem__(self, index):
# #         p_ratio = random.random()
# #         if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
# #             img, mask = self.load_img_and_mask(index)
# #             if self.transform:
# #                 img, mask = self.transform(img, mask)
# #         else:
# #             img, mask = self.load_mosaic_img_and_mask(index)
# #             if self.transform:
# #                 img, mask = self.transform(img, mask)
# #
# #         img = torch.from_numpy(img).permute(2, 0, 1).float()
# #         mask = torch.from_numpy(mask).long()
# #         img_id = self.img_ids[index]
# #         results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
# #         return results
# #
# #     def __len__(self):
# #         return len(self.img_ids)
# #
# #     def get_img_ids(self, data_root, img_dir, mask_dir):
# #         img_filename_list = os.listdir(osp.join(data_root, img_dir))
# #         mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
# #         assert len(img_filename_list) == len(mask_filename_list)
# #         img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
# #         return img_ids
# #
# #     def load_img_and_mask(self, index):
# #         img_id = self.img_ids[index]
# #         img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
# #         mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
# #
# #         # ---- 修改点：使用 rasterio 读取4通道图像 ----
# #         with rasterio.open(img_name) as src:
# #             img = src.read()          # shape: (bands, H, W)，bands=4
# #         img = np.transpose(img, (1, 2, 0))  # 转为 (H, W, bands)
# #         # 假设图像数据为 uint8，如果不是，请根据实际数据类型转换（例如除以65535后乘以255）
# #         img = img.astype(np.uint8)
# #         # -----------------------------------------
# #
# #         # ---- 掩码读取保持不变（标签为RGB三通道） ----
# #         mask_rgb = np.array(Image.open(mask_name).convert('RGB'))
# #         h, w = mask_rgb.shape[:2]
# #         mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
# #         for color, idx in self.color2index.items():
# #             mask[(mask_rgb == color).all(axis=2)] = idx
# #
# #         # 将 img 和 mask 转换为 PIL Image 以保持原有接口（4通道图像PIL可处理为RGBA）
# #         img = Image.fromarray(img)
# #         mask = Image.fromarray(mask, mode='L')
# #         return img, mask
# #
# #     def load_mosaic_img_and_mask(self, index):
# #         # 马赛克增强，最终图像尺寸为 self.img_size = (256, 256)
# #         indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
# #         img_a, mask_a = self.load_img_and_mask(indexes[0])
# #         img_b, mask_b = self.load_img_and_mask(indexes[1])
# #         img_c, mask_c = self.load_img_and_mask(indexes[2])
# #         img_d, mask_d = self.load_img_and_mask(indexes[3])
# #
# #         img_a, mask_a = np.array(img_a), np.array(mask_a)
# #         img_b, mask_b = np.array(img_b), np.array(mask_b)
# #         img_c, mask_c = np.array(img_c), np.array(mask_c)
# #         img_d, mask_d = np.array(img_d), np.array(mask_d)
# #
# #         w = self.img_size[1]
# #         h = self.img_size[0]
# #
# #         start_x = w // 4
# #         strat_y = h // 4
# #         offset_x = random.randint(start_x, (w - start_x))
# #         offset_y = random.randint(strat_y, (h - strat_y))
# #
# #         crop_size_a = (offset_x, offset_y)
# #         crop_size_b = (w - offset_x, offset_y)
# #         crop_size_c = (offset_x, h - offset_y)
# #         crop_size_d = (w - offset_x, h - offset_y)
# #
# #         random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
# #         random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
# #         random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
# #         random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])
# #
# #         croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
# #         croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
# #         croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
# #         croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())
# #
# #         img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
# #         img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
# #         img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
# #         img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']
# #
# #         top = np.concatenate((img_crop_a, img_crop_b), axis=1)
# #         bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
# #         img = np.concatenate((top, bottom), axis=0)
# #
# #         top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
# #         bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
# #         mask = np.concatenate((top_mask, bottom_mask), axis=0)
# #         mask = np.ascontiguousarray(mask)
# #         img = np.ascontiguousarray(img)
# #
# #         img = Image.fromarray(img)
# #         mask = Image.fromarray(mask)
# #         return img, mask

#三波段
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from .transform import *   # 请确保此模块包含 Compose, RandomScale, SmartCropV1
import matplotlib.patches as mpatches
from PIL import Image
import random

# -------------------- 修改1：尺寸常量设为 256 --------------------
ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)
# ----------------------------------------------------------------

# -------------------- 自定义类别和颜色映射 --------------------
CLASSES = ('water', 'vegetation', 'road', 'bareland', 'buildings')
PALETTE = [
    [0, 0, 255],       # 水体 - 蓝色
    [0, 128, 0],       # 植被 - 绿色
    [0, 255, 255],     # 道路 - 青色
    [153, 99, 0],      # 裸地 - 褐色
    [255, 0, 0]        # city - 红色
]
IGNORE_INDEX = len(CLASSES)   # 忽略索引设为 5
# ----------------------------------------------------------------

def get_training_transform():
    mean = [0.485, 0.456, 0.406, 0.5]   # RGB + NIR 示例值
    std  = [0.229, 0.224, 0.225, 0.2]
    train_transform = [
        albu.RandomRotate90(p=0.25),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize(mean=mean, std=std)
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # 统一转换为 numpy 数组（便于 albumentations 处理）
    img = np.array(img)  # (H, W, 4)
    mask = np.array(mask)  # (H, W)

    # 先缩放到 256x256（保持宽高比，再中心裁剪或直接 resize）
    transform = albu.Compose([
        albu.Resize(height=256, width=256),  # 强制缩放至 256x256
        albu.RandomRotate90(p=0.25),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
    ])
    augmented = transform(image=img, mask=mask)
    img, mask = augmented['image'], augmented['mask']
    return img, mask

def get_val_transform():
    mean = [0.485, 0.456, 0.406, 0.5]
    std  = [0.229, 0.224, 0.225, 0.2]
    val_transform = [albu.Normalize(mean=mean, std=std)]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    transform = albu.Compose([
        albu.Resize(height=256, width=256),
        albu.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
    ])
    augmented = transform(image=img, mask=mask)
    return augmented['image'], augmented['mask']

def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)

def test_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_test_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

class LYY(Dataset):
    def __init__(self, data_root=r'E:\sjj\LYY\val', mode='val', img_dir='images', mask_dir='labels',
                 img_suffix='.tif', mask_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size    # 此时 img_size 为 (256, 256)
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
        # 构建颜色到索引的映射字典
        self.color2index = self._build_color2index()

    def _build_color2index(self):
        color2index = {}
        for idx, color in enumerate(PALETTE):
            color2index[tuple(color)] = idx
        return color2index

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
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

        # 使用 tifffile 读取 4 波段图像 (H, W, 4)
        import tifffile
        img_array = tifffile.imread(img_name)
        # 转换为 uint8（假设原始数据为 0-65535 或 0-255，需根据实际调整）
        if img_array.dtype != np.uint8:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        # 转换为 PIL Image，模式 'RGBA'（4 通道）
        img = Image.fromarray(img_array, mode='RGBA')

        # 读取标签（RGB 彩色图转索引图）
        mask_rgb = np.array(Image.open(mask_name).convert('RGB'))
        h, w = mask_rgb.shape[:2]
        mask_array = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
        for color, idx in self.color2index.items():
            mask_array[(mask_rgb == color).all(axis=2)] = idx
        mask = Image.fromarray(mask_array, mode='L')
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        # 马赛克增强，最终图像尺寸为 self.img_size = (256, 256)
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