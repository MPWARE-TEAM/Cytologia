import numpy as np
import albumentations as A
import random
import cv2
import staintools
import torchstain
from PIL import Image
from torchvision import transforms

from albumentations import HistogramMatching
from albumentations.augmentations.domain_adaptation import apply_histogram


class HistogramMatchingPerClass(HistogramMatching):
    def apply(self, img, reference_image=None, blend_ratio=0.5, **params):
        # Pick image with respect to class
        reference_image = self.read_fn(random.choice(self.reference_images[params["class_name"]]))
        return apply_histogram(img, reference_image, blend_ratio)

    def get_params(self):
        return {
            "reference_image": None,
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        params.update({"class_name": kwargs["class_name"]})
        return params


class Stainer(A.DualTransform):

    def __init__(self, ref_images, method, luminosity=True, always_apply=False, p=1.0):
        super(Stainer, self).__init__(always_apply=always_apply, p=p)
        self.luminosity = luminosity
        self.method = method
        self.stain_normalizer = []
        self.torchstain_T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        if ref_images is None:
            # Generic reference
            if method == 'macenko':
                stain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
                self.stain_normalizer.append(stain_normalizer)
        else:
            for ref_image in ref_images:
                ref_image = np.array(Image.open(ref_image))
                if method == 'reinhard':
                    stain_normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
                    stain_normalizer.fit(self.torchstain_T(ref_image))
                    self.stain_normalizer.append(stain_normalizer)
                elif method == 'macenko':
                    stain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
                    stain_normalizer.fit(self.torchstain_T(ref_image))
                    self.stain_normalizer.append(stain_normalizer)
                else:
                    # vahadane
                    ref_image = staintools.LuminosityStandardizer.standardize(ref_image) if self.luminosity == True else ref_image
                    stain_normalizer = staintools.StainNormalizer(method=method)
                    stain_normalizer.fit(ref_image)
                    self.stain_normalizer.append(stain_normalizer)

    def apply(self, img, **params):
        # Standardize brightness (optional, can improve the tissue mask calculation)
        if self.luminosity == True:
            img = staintools.LuminosityStandardizer.standardize(img)
        if self.method == 'macenko':
            stain_normalizer = self.stain_normalizer[0]
            img, _, _ = stain_normalizer.normalize(I=self.torchstain_T(img), stains=False)
            img = img.contiguous().cpu().numpy().astype(np.uint8)
        elif self.method == 'reinhard':
            stain_normalizer = np.random.choice(self.stain_normalizer, 1)[0]
            img = stain_normalizer.normalize(I=self.torchstain_T(img))
            img = img.contiguous().cpu().numpy().astype(np.uint8)
        else:
            stain_normalizer = np.random.choice(self.stain_normalizer, 1)[0]
            img = stain_normalizer.transform(img)
        return img

    # def get_transform_init_args_names(self):
    #     return ("ref_images", "method", "luminosity")

