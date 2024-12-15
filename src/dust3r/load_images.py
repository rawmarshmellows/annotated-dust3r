# image_loader.py

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2  # noqa: E402
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Define image normalization transform
IMG_NORMALIZATION = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


@dataclass
class ImageData:
    img_tensor: torch.Tensor
    true_shape: np.ndarray
    idx: int
    instance: str

    def __post_init__(self):
        self.validate_img_tensor()
        self.validate_true_shape()
        self.validate_idx()
        self.validate_instance()

    def validate_img_tensor(self):
        if not isinstance(self.img_tensor, torch.Tensor):
            raise TypeError("img_tensor must be a torch.Tensor")
        if self.img_tensor.ndim != 4 or self.img_tensor.size(0) != 1:
            raise ValueError("img_tensor must have shape (1, 3, H, W)")

    def validate_true_shape(self):
        if not isinstance(self.true_shape, np.ndarray):
            raise TypeError("true_shape must be a numpy.ndarray")
        if self.true_shape.shape != (1, 2):
            raise ValueError("true_shape must have shape (1, 2)")
        if self.true_shape.dtype != np.int32:
            raise TypeError("true_shape must be of dtype int32")

    def validate_idx(self):
        if not isinstance(self.idx, int):
            raise TypeError("idx must be an integer")

    def validate_instance(self):
        if not isinstance(self.instance, str):
            raise TypeError("instance must be a string")

    def to_dict(self):
        return {"img": self.img_tensor, "true_shape": self.true_shape, "idx": self.idx, "instance": self.instance}


@dataclass
class LoadConfig:
    size: int
    square_ok: bool = False
    verbose: bool = True

    def __post_init__(self):
        self.validate_size()
        self.validate_square_ok()
        self.validate_verbose()

    def validate_size(self):
        if not isinstance(self.size, int):
            raise TypeError("size must be an integer")
        if self.size <= 0:
            raise ValueError("size must be a positive integer")

    def validate_square_ok(self):
        if not isinstance(self.square_ok, bool):
            raise TypeError("square_ok must be a boolean")

    def validate_verbose(self):
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean")


def convert_to_rgb(
    tensor: Union[torch.Tensor, List[torch.Tensor], np.ndarray], true_shape: Optional[Tuple[int, int]] = None
) -> Union[torch.Tensor, List[torch.Tensor], np.ndarray]:
    """
    Convert tensors or arrays to RGB format with normalization.

    Args:
        tensor (torch.Tensor | List[torch.Tensor] | np.ndarray): Input image data.
        true_shape (tuple, optional): Desired shape to crop the image.

    Returns:
        torch.Tensor | List[torch.Tensor] | np.ndarray: Normalized RGB image(s).

    Raises:
        TypeError: If tensor is not a torch.Tensor, list of torch.Tensor, or numpy.ndarray.
    """
    if isinstance(tensor, list):
        return [convert_to_rgb(x, true_shape=true_shape) for x in tensor]

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)
    elif tensor.ndim == 4 and tensor.shape[1] == 3:
        tensor = tensor.transpose(0, 2, 3, 1)

    if true_shape:
        H, W = true_shape
        tensor = tensor[:H, :W]

    if tensor.dtype == np.uint8:
        img = tensor.astype(np.float32) / 255.0
    else:
        img = (tensor * 0.5) + 0.5

    return np.clip(img, 0, 1)


def resize_pil_image(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize a PIL Image based on the target size.

    Args:
        img (PIL.Image.Image): Image to resize.
        target_size (int): Target size for the longer edge.

    Returns:
        PIL.Image.Image: Resized image.
    """
    original_size = max(img.size)
    resample_method = Image.LANCZOS if original_size > target_size else Image.BICUBIC
    new_size = tuple(int(round(dim * target_size / original_size)) for dim in img.size)
    return img.resize(new_size, resample_method)


def crop_pil_image(img: Image.Image, target_size: int, square_ok: bool) -> Image.Image:
    """
    Crop the PIL Image to the target size.

    Args:
        img (PIL.Image.Image): Image to crop.
        target_size (int): Desired size after cropping.
        square_ok (bool): Allow non-square cropping.

    Returns:
        PIL.Image.Image: Cropped image.

    Raises:
        ValueError: If the target size is larger than the image dimensions.
    """
    width, height = img.size
    center_x, center_y = width // 2, height // 2

    if target_size > width or target_size > height:
        raise ValueError("target_size cannot be larger than image dimensions")

    box = calculate_crop_box(width, height, center_x, center_y, target_size, square_ok)
    return img.crop(box)


def calculate_crop_box(
    width: int, height: int, center_x: int, center_y: int, target_size: int, square_ok: bool
) -> Tuple[int, int, int, int]:
    """
    Calculate the crop box for the image.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        center_x (int): X-coordinate of the center.
        center_y (int): Y-coordinate of the center.
        target_size (int): Desired crop size.
        square_ok (bool): Allow non-square cropping.

    Returns:
        Tuple[int, int, int, int]: Crop box coordinates.
    """
    if target_size == 224:
        half = target_size // 2
        return (center_x - half, center_y - half, center_x + half, center_y + half)
    else:
        # Modular loop logic: separate calculation
        half_width = (2 * center_x // 16) * 8
        half_height = (2 * center_y // 16) * 8
        if not square_ok and width == height:
            half_height = 3 * half_width // 4
        return (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)


def load_images(image_paths: List[str], config: LoadConfig) -> List[ImageData]:
    """
    Load and preprocess images from a list of file paths.

    Args:
        image_paths (List[str]): List of image file paths.
        config (LoadConfig): Configuration for loading images.

    Returns:
        List[ImageData]: List of preprocessed image data.

    Raises:
        ValueError: If no valid images are loaded.
        TypeError: If image_paths is not a list of strings.
    """
    validate_image_paths(image_paths)
    if config.verbose:
        logger.info(f">> Loading {len(image_paths)} images.")

    images: List[ImageData] = []
    for path in image_paths:
        if not path.lower().endswith(tuple(SUPPORTED_IMAGE_EXTENSIONS)):
            logger.warning(f"Skipping unsupported file format: {path}")
            continue

        try:
            img = open_and_prepare_image(path)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

        try:
            img = resize_image_if_needed(img, config.size)
            # img = crop_pil_image(img, config.size, config.square_ok)
        except ValueError as ve:
            logger.warning(f"Skipping {path}: {ve}")
            continue

        resized_size = img.size
        if config.verbose:
            logger.info(f" - Added {path} with resolution {img.size[0]}x{img.size[1]}")

        try:
            img_tensor, true_shape = prepare_image_tensor(img)
            image_data = ImageData(
                img_tensor=img_tensor, true_shape=true_shape, idx=len(images), instance=str(len(images))
            )
            images.append(image_data)
        except Exception as e:
            logger.warning(f"Failed to create ImageData for {path}: {e}")

    if not images:
        raise ValueError("No valid images were loaded.")

    if config.verbose:
        logger.info(f" (Successfully loaded {len(images)} images)")

    return images


def validate_image_paths(image_paths: List[str]):
    """
    Validate that image_paths is a list of strings.

    Args:
        image_paths (List[str]): List of image file paths.

    Raises:
        TypeError: If image_paths is not a list or contains non-string elements.
    """
    if not isinstance(image_paths, list):
        raise TypeError("image_paths must be a list of strings.")
    for path in image_paths:
        if not isinstance(path, str):
            raise TypeError("All items in image_paths must be strings.")


def open_and_prepare_image(path: str) -> Image.Image:
    """
    Open an image and prepare it for processing.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Prepared image.

    Raises:
        IOError: If the image cannot be opened.
    """
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def resize_image_if_needed(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize the image based on the target size.

    Args:
        img (PIL.Image.Image): Image to resize.
        target_size (int): Target size for resizing.

    Returns:
        PIL.Image.Image: Resized image.
    """
    original_size = max(img.size)
    resample_method = Image.LANCZOS if original_size > target_size else Image.BICUBIC
    new_size = tuple(int(round(dim * target_size / original_size)) for dim in img.size)
    return img.resize(new_size, resample_method)


def prepare_image_tensor(img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Prepare the image tensor and true shape.

    Args:
        img (PIL.Image.Image): Image to convert.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: Image tensor and true shape.
    """
    img_tensor = IMG_NORMALIZATION(img).unsqueeze(0)
    true_shape = np.array([img.size[::-1]], dtype=np.int32)
    return img_tensor, true_shape
