import os, glob, random
from collections import Counter
from PIL import Image
from math import isqrt, ceil
from typing import List
import logging
import hashlib
import torch
from safetensors.torch import save_file, safe_open
from insightface.app.common import Face

from modules.images import FilenameGenerator, get_next_sequence_number
from modules import shared, script_callbacks
from scripts.reactor_globals import DEVICE, BASE_PATH, FACE_MODELS_PATH, IS_SDNEXT

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")

MODELS_PATH = None

def set_Device(value):
    global DEVICE
    DEVICE = value
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(DEVICE)

def get_Device():
    global DEVICE
    return DEVICE

def set_SDNEXT():
    global IS_SDNEXT
    IS_SDNEXT = True

def get_SDNEXT():
    global IS_SDNEXT
    return IS_SDNEXT

def make_grid(image_list: List):
    
    # Count the occurrences of each image size in the image_list
    size_counter = Counter(image.size for image in image_list)
    
    # Get the most common image size (size with the highest count)
    common_size = size_counter.most_common(1)[0][0]
    
    # Filter the image_list to include only images with the common size
    image_list = [image for image in image_list if image.size == common_size]
    
    # Get the dimensions (width and height) of the common size
    size = common_size
    
    # If there are more than one image in the image_list
    if len(image_list) > 1:
        num_images = len(image_list)
        
        # Calculate the number of rows and columns for the grid
        rows = isqrt(num_images)
        cols = ceil(num_images / rows)

        # Calculate the size of the square image
        square_size = (cols * size[0], rows * size[1])

        # Create a new RGB image with the square size
        square_image = Image.new("RGB", square_size)

        # Paste each image onto the square image at the appropriate position
        for i, image in enumerate(image_list):
            row = i // cols
            col = i % cols

            square_image.paste(image, (col * size[0], row * size[1]))

        # Return the resulting square image
        return square_image
    
    # Return None if there are no images or only one image in the image_list
    return None

def get_image_path(image, path, basename, seed=None, prompt=None, extension='png', p=None, suffix=""):
    
    namegen = FilenameGenerator(p, seed, prompt, image)

    save_to_dirs = shared.opts.save_to_dirs

    if save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if seed is None:
        file_decoration = ""
    elif shared.opts.save_to_dirs:
        file_decoration = shared.opts.samples_filename_pattern or "[seed]"
    else:
        file_decoration = shared.opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

    file_decoration = namegen.apply(file_decoration) + suffix

    add_number = shared.opts.save_images_add_number or file_decoration == ''

    if file_decoration != "" and add_number:
        file_decoration = f"-{file_decoration}"

    if add_number:
        basecount = get_next_sequence_number(path, basename)
        fullfn = None
        for i in range(500):
            fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
            fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
            if not os.path.exists(fullfn):
                break
    else:
        fullfn = os.path.join(path, f"{file_decoration}.{extension}")

    pnginfo = {}

    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    # script_callbacks.before_image_saved_callback(params)

    fullfn = params.filename

    fullfn_without_extension, extension = os.path.splitext(params.filename)
    if hasattr(os, 'statvfs'):
        max_name_len = os.statvfs(path).f_namemax
        fullfn_without_extension = fullfn_without_extension[:max_name_len - max(4, len(extension))]
        params.filename = fullfn_without_extension + extension
        fullfn = params.filename

    return fullfn

def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def get_image_md5hash(image: Image.Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()

def save_face_model(face: Face, filename: str) -> None:
    try:
        tensors = {
            "bbox": torch.tensor(face["bbox"]),
            "kps": torch.tensor(face["kps"]),
            "det_score": torch.tensor(face["det_score"]),
            "landmark_3d_68": torch.tensor(face["landmark_3d_68"]),
            "pose": torch.tensor(face["pose"]),
            "landmark_2d_106": torch.tensor(face["landmark_2d_106"]),
            "embedding": torch.tensor(face["embedding"]),
            "gender": torch.tensor(face["gender"]),
            "age": torch.tensor(face["age"]),
        }
        save_file(tensors, filename)
        # print(f"Face model has been saved to '{filename}'")
    except Exception as e:
        print(f"Error: {e}")

def get_models():
    global MODELS_PATH
    models_path_init = os.path.join(models_path, "insightface/**/*")
    models = glob.glob(models_path_init, recursive=True)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    model_inswapper = os.path.join(models_path, "insightface/inswapper_128.onnx")
    model_inswapper_lost = [path for path in models if 'inswapper_128.onnx' in path and  os.path.split(path)[0].split('\\')[-1] != "insightface"]
    if model_inswapper_lost and not os.path.exists(model_inswapper):
        os.rename(model_inswapper_lost, model_inswapper)
    models_names = []
    for model in models:
        model_path = os.path.split(model)
        if MODELS_PATH is None:
            MODELS_PATH = model_path[0]
        model_name = model_path[1]
        models_names.append(model_name)
    return models_names

def load_face_model(filename: str):
    face = {}
    model_path = os.path.join(FACE_MODELS_PATH, filename)
    with safe_open(model_path, framework="pt") as f:
        for k in f.keys():
            face[k] = f.get_tensor(k).numpy()
    return Face(face)

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_model_names(get_models):
    models = get_models()
    names = ["None"]
    for x in models:
        names.append(os.path.basename(x))
    return names

def get_images_from_folder(path: str):
    files_path = os.path.join(path, "*")
    files = glob.glob(files_path)
    images = []
    images_names = []
    for x in files:
        if x.endswith(('jpg', 'png', 'jpeg', 'webp', 'bmp')):
            images.append(Image.open(x))
            images_names.append(os.path.basename(x))
    return images,images_names
    # return [Image.open(x) for x in images if x.endswith(('jpg', 'png', 'jpeg', 'webp', 'bmp'))],[os.path.basename(x) for x in images if x.endswith(('jpg', 'png', 'jpeg', 'webp', 'bmp'))]

def get_random_image_from_folder(path: str):
    images,names = get_images_from_folder(path)
    random_image_index = random.randint(0, len(images) - 1)
    return [images[random_image_index]],[names[random_image_index]]

def get_images_from_list(imgs: List):
    images = []
    images_names = []
    for x in imgs:
        images.append(Image.open(os.path.abspath(x.name)))
        images_names.append(os.path.basename(x.name))
    return images,images_names
    # return [Image.open(os.path.abspath(x.name)) for x in imgs],[os.path.basename(x.name) for x in imgs]
