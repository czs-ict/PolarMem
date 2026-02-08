import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional

class OverlapPatcher:
    """
    负责将图像切分为重叠的 Patch，并计算归一化坐标 (0-1000)。
    """
    def __init__(self, patch_size: int = 448, stride_ratio: float = 0.5):
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_ratio)

    def process(
        self,
        image: Union[Image.Image, np.ndarray],
        max_patches: Optional[int] = None,
    ) -> List[Tuple[Image.Image, List[int]]]:
        """
        输入: Numpy Array 或 PIL Image
        输出: List of (image_patch, [x1, y1, x2, y2])
        坐标已归一化到 0-1000。
        """
        # 统一转换为 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        patches = []

        # 简单的滑动窗口逻辑
        # 如果图片小于 patch_size，直接返回原图 + [0,0,1000,1000]
        if width <= self.patch_size and height <= self.patch_size:
            return [(image, [0, 0, 1000, 1000])]

        # 生成网格坐标
        # x_steps: 0, stride, 2*stride, ... 直到右边缘被覆盖
        x_coords = list(range(0, width - self.patch_size + 1, self.stride))
        if (width - self.patch_size) % self.stride != 0:
            x_coords.append(width - self.patch_size) # 补上最后一个贴边的
            
        y_coords = list(range(0, height - self.patch_size + 1, self.stride))
        if (height - self.patch_size) % self.stride != 0:
            y_coords.append(height - self.patch_size)

        # 极端情况处理：如果图片某个维度本身就比 patch_size 小
        if width < self.patch_size:
            x_coords = [0]
        if height < self.patch_size:
            y_coords = [0]
        
        # 去重并排序
        x_coords = sorted(list(set(x_coords)))
        y_coords = sorted(list(set(y_coords)))

        for y in y_coords:
            for x in x_coords:
                # 实际裁切区域
                real_x1 = x
                real_y1 = y
                real_x2 = min(x + self.patch_size, width)
                real_y2 = min(y + self.patch_size, height)

                patch = image.crop((real_x1, real_y1, real_x2, real_y2))
                
                # 归一化坐标计算 (0-1000)
                norm_x1 = int((real_x1 / width) * 1000)
                norm_y1 = int((real_y1 / height) * 1000)
                norm_x2 = int((real_x2 / width) * 1000)
                norm_y2 = int((real_y2 / height) * 1000)

                patches.append((patch, [norm_x1, norm_y1, norm_x2, norm_y2]))

                if max_patches is not None and len(patches) >= max_patches:
                    return patches

        return patches
