import PIL.Image
import io
import base64
import logging

def compress_image(image:PIL.Image.Image, target_size) -> None:
    '''
    压缩图像至指定大小

    :param image: 待压缩的图像。该图像将被原地压缩
    :param target_size: 目标大小，单位为字节

    :raises ValueError: 如果图像格式不受支持或无法识别
    '''
    fmt = image.format
    if fmt is None:
        raise ValueError("Image format not recognized")
    if fmt not in ["JPEG", "PNG"]:
        raise ValueError(f"Unsupported image format: {fmt}")
    
    while True:
        w,h = image.size
        img_size = len(image.tobytes())
        scale = target_size / img_size
        if scale > 0.9:
            scale = 0.9
        if scale < 0.5:
            scale = 0.5
        image.thumbnail((w*scale, h*scale))
        img_size = len(image.tobytes())
        if img_size <= target_size:
            break

def thumbnail_url(image: PIL.Image.Image, target_size) -> str:
    '''
    压缩图像并生成base64编码的URL

    :param image: 待处理的图片。该图像将被原地压缩
    :param target_size: 目标大小，单位为字节

    :return: 图片URL
    
    :raises ValueError: 如果图像格式不受支持或无法识别
    '''
    fmt = image.format
    if fmt is None:
        raise ValueError("Image format not recognized")
    if fmt not in ["JPEG", "PNG"]:
        raise ValueError(f"Unsupported image format: {fmt}")

    compress_image(image, target_size)

    with io.BytesIO() as buf:
        image.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"
