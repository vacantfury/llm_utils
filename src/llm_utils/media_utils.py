"""
Shared image encoding utilities for LLM service implementations.
"""
import base64
import io
from pathlib import Path
from typing import Any, Tuple


_FORMAT_TO_MIME = {
    'JPEG': 'image/jpeg',
    'JPG': 'image/jpeg',
    'PNG': 'image/png',
    'GIF': 'image/gif',
    'WEBP': 'image/webp',
}

_EXT_TO_MIME = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
}


def encode_image_to_b64(image: Any) -> Tuple[str, str]:
    """
    Encode an image to base64 and determine its MIME type.

    Args:
        image: Either a file path (str/Path) or a PIL Image object

    Returns:
        (base64_data, mime_type) tuple
    """
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        buffer = io.BytesIO()
        img_format = image.format or 'PNG'
        mime_type = _FORMAT_TO_MIME.get(img_format.upper(), 'image/png')
        save_format = 'JPEG' if img_format.upper() == 'JPG' else img_format
        image.save(buffer, format=save_format)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_data, mime_type

    # File path
    image_path = str(image)
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    ext = Path(image_path).suffix.lower()
    mime_type = _EXT_TO_MIME.get(ext, 'image/jpeg')
    return image_data, mime_type
