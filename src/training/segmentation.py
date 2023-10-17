from open_clip.model import PACL
from open_clip.loss import PACLLoss
from open_clip.transform import image_transform
from open_clip import get_tokenizer

from PIL import Image
import torch.nn.functional as F


LABELS = [
    'A photo of an airplane',
    'A photo of a dog',
    'A photo a wooden chair',
    'A photo of a background'
]

COLORS = [
    [220,20,60],
    [135,206,235],
    [50,205,50],
    [0, 0, 0]
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

    print(class_embeddings.shape)
    
    patch_similarity = img @ class_embeddings

    print(patch_similarity)
