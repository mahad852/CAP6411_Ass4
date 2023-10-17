from open_clip.model import PACL
from open_clip.transform import image_transform
from open_clip import get_tokenizer

from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
import os

from torchvision.transforms import InterpolationMode, Resize, CenterCrop

LABELS = [
    'A photo of an airplane',
    'A photo of a dog',
    'A photo a wooden chair',
    'A photo of a background',
    'A photo of a cat',
]

COLORS = [
    (220,20,60), # red -airplane
    (135,206,235), # blue - dog
    (50,205,50), # green - chair
    (0, 0, 0), # background
    (255,255,0) #yellow - cat
]

def load_image(img_path: str, model: PACL):
    image = Image.open(img_path)

    image_mean = getattr(model.visual, 'image_mean', None)
    image_std = getattr(model.visual, 'image_std', None)
    
    transform = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )
    image = transform(image)
    return image.unsqueeze(0)

def get_original_image(img_path: str, model: PACL):
    image = Image.open(img_path)
    original_converter = T.Compose([
        Resize(model.visual.image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(model.visual.image_size)
    ])
    return original_converter(image)

def get_all_labels():
    return [
        'This is an image of airplane',
        'This is the background'
    ]

def perform_segmentation(img_path: str, model: PACL, model_name: str):
    img = load_image(img_path, model).to('cuda')
    
    tokenizer = get_tokenizer(model_name)
    texts = tokenizer(LABELS).to('cuda')
    class_embeddings = F.normalize(model.encode_text(texts), dim=-1)
    class_embeddings = class_embeddings.view(4, -1)
    class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
    class_embeddings = class_embeddings.T

    output = model(image=img)
    image_features = output['image_features'] if isinstance(output, dict) else output[0]
    
    patch_similarity = image_features @ class_embeddings
    patch_similarity = F.softmax(patch_similarity[0], dim = 1).argmax(dim = 1)
    
    img_original = get_original_image(img_path, model)

    for i_patch in range(len(patch_similarity)):
        color = COLORS[patch_similarity[i_patch].item()]

        row_index = int(i_patch / 14) * 16
        col_index = (i_patch % 14) * 16

        for r in range(row_index, row_index + 16):
            for c in range(col_index, col_index + 16):
                img_original.putpixel((r, c), color)
    

    new_img_path = os.path.join(os.sep.join(img_path.split(os.sep)[:-1]), 'seg.png')
    img_original.save(new_img_path)
