import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def add_noisy_pixels(image, noise_intensity=10, noise_fraction=0.05):
    # Create a copy of the original image to apply noise
    noisy_image = image.copy()

    # Get image dimensions
    height, width, _ = image.shape

    # Determine the number of pixels to modify
    num_pixels = int(noise_fraction * height * width)

    # Randomly select pixel indices to modify
    y_coords = np.random.randint(0, height, num_pixels)
    x_coords = np.random.randint(0, width, num_pixels)

    # Generate noise values within the specified intensity range
    noise = np.random.randint(-noise_intensity, noise_intensity + 1, (num_pixels, 3))

    # Apply noise to the selected pixels
    for i in range(num_pixels):
        y, x = y_coords[i], x_coords[i]
        noisy_image[y, x] = np.clip(noisy_image[y, x] + noise[i], 0, 255)

    return noisy_image


def apply_otsu(image):
    # Convert RGB image to greyscale
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Find Otsu's threshold
    threshold = threshold_otsu(np.array(grey).flatten())
    mask = (grey < threshold).astype(bool)
    return mask


def remove_wbc(img, x1, y1, x2, y2, bg_thr=150, kernel_size_color=(20, 20), kernel_size=(7, 7), noise_intensity=8,
               noise_fraction=0.10, verbose=False):
    mask = apply_otsu(img)

    # Dilate full mask prior to compute average bg color
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_color)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask_inv = np.where(mask_dilated == True, 0, 1)
    bg_full = np.zeros_like(img, dtype=np.uint8)
    bg_full[:, :, 0] = img[:, :, 0] * mask_inv.astype(np.uint8)
    bg_full[:, :, 1] = img[:, :, 1] * mask_inv.astype(np.uint8)
    bg_full[:, :, 2] = img[:, :, 2] * mask_inv.astype(np.uint8)
    red = bg_full[:, :, 0]
    green = bg_full[:, :, 1]
    blue = bg_full[:, :, 2]
    fill_color = (np.mean(red[red > bg_thr]), np.mean(green[green > bg_thr]), np.mean(blue[blue > bg_thr]))
    if verbose:
        d = plt.imshow(bg_full)
        d = plt.show()

    # Crop
    part_img = img[y1:y2, x1:x2]
    part_mask = mask[y1:y2, x1:x2].astype(np.uint8)

    if verbose:
        d = plt.imshow(part_mask)
        d = plt.show()

    # Find the largest contour inside the crop
    part_mask_largest = part_mask.copy()
    contours, hierarchy = cv2.findContours(part_mask_largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    part_mask_largest = cv2.drawContours(part_mask_largest, [largest_contour], -1, 255, thickness=cv2.FILLED)
    part_mask_largest[part_mask_largest != 255] = 0
    part_mask_largest[part_mask_largest != 0] = 1
    if verbose:
        d = plt.imshow(part_mask_largest)
        d = plt.show()

    # Dilate contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    part_mask_largest = cv2.dilate(part_mask_largest, kernel, iterations=1)
    if verbose:
        d = plt.imshow(part_mask_largest)
        d = plt.show()
        d = plt.imshow(part_img)
        d = plt.show()

    # Masked data
    bg = np.zeros_like(part_img, dtype=np.uint8)
    bg[:, :, 0] = part_img[:, :, 0] * part_mask_largest
    bg[:, :, 1] = part_img[:, :, 1] * part_mask_largest
    bg[:, :, 2] = part_img[:, :, 2] * part_mask_largest
    if verbose:
        d = plt.imshow(bg)
        d = plt.show()

    # Fill color
    bg_color = np.zeros_like(part_img, dtype=np.uint8)
    bg_color[:, :, 0] = fill_color[0]
    bg_color[:, :, 1] = fill_color[1]
    bg_color[:, :, 2] = fill_color[2]
    bg_color[:, :, 0] = bg_color[:, :, 0] * part_mask_largest
    bg_color[:, :, 1] = bg_color[:, :, 1] * part_mask_largest
    bg_color[:, :, 2] = bg_color[:, :, 2] * part_mask_largest
    if verbose:
        d = plt.imshow(bg_color)
        d = plt.show()
        d = plt.imshow(part_img - bg)
        d = plt.show()

    img_bg = img.copy()
    img_bg[y1:y2, x1:x2] = part_img - bg
    if verbose:
        d = plt.imshow(img_bg)
        d = plt.show()

    bg_color_noisy = add_noisy_pixels(bg_color, noise_intensity=noise_intensity, noise_fraction=noise_fraction)
    bg_color_noisy[:, :, 0] = bg_color_noisy[:, :, 0] * part_mask_largest
    bg_color_noisy[:, :, 1] = bg_color_noisy[:, :, 1] * part_mask_largest
    bg_color_noisy[:, :, 2] = bg_color_noisy[:, :, 2] * part_mask_largest
    if verbose:
        d = plt.imshow(bg_color_noisy)
        d = plt.show()

        # Final image
    img_bg[y1:y2, x1:x2] = img_bg[y1:y2, x1:x2] + bg_color_noisy
    if verbose:
        d = plt.imshow(img_bg)
        d = plt.grid(False)

    return img_bg
