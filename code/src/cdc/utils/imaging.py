import numpy as np
import random
from PIL import Image
import cv2
from typing import Any
from albumentations.augmentations.crops import functional as fcrops
from albumentations.core.transforms_interface import DualTransform


# https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
def image_colorfulness(image: np.ndarray) -> float:
    """
    Args:
        image: a single image in RGB format

    Returns:
        float: colorfulness
    """
    r, g, b = np.rollaxis(image.astype(float), 2)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rb_mean = np.mean(rg)
    rb_std = np.std(rg)
    yb_mean = np.mean(yb)
    yb_std = np.std(yb)
    std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))
    return std_root + (0.3 * mean_root)


def is_valid_jpeg(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Check for corruption
        return True  # If no exception, the file is valid
    except Exception as e:
        print(f"Invalid image file: {file_path}, Error: {e}")
        return False


# Extract random crop from image (no resize)
class RandomUnsizedCrop(DualTransform):

    def __init__(
        self,
        min_max_height: tuple[int, int],
        h_w_location: tuple[int, int] | None = None,
        *,
        w2h_ratio: float = 1.0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(
            p=p,
            always_apply=always_apply
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio
        self.h_w_location = h_w_location

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        crop = fcrops.crop(img, *crop_coords)
        return crop

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]

        crop_height = self.py_random.randint(*self.min_max_height)
        crop_width = int(crop_height * self.w2h_ratio)

        crop_shape = (crop_height, crop_width)

        if self.h_w_location is None:
            h_start = self.py_random.random()
            w_start = self.py_random.random()
        else:
            h_start, w_start = self.h_w_location

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("min_max_height", "w2h_ratio", "h_w_location")
