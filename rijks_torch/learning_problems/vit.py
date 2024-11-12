from .defaults import freezeLayers
from torch import nn
from torchvision import models
from torchvision.models.vision_transformer import *

def get_vit_b_16_problem(off_the_shelf: bool, dl):
    """
    Returns the whole problem statement for training vit_b_16 on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    print("retrieving model vit_b_16")
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.heads.head = nn.Linear(768, len(dl.materials))

    return model, dl


def get_vit_b_16_drop_problem(off_the_shelf: bool, dl):
    """ Same but with a dropout layer. This version is used for fine tuning """
    model, dl = get_vit_b_16_problem(off_the_shelf, dl)
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(768, len(dl.materials))
    )
    return model, dl
