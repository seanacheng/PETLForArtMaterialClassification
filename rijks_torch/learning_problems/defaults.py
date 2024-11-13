from torchvision import transforms
from torch import nn

def freezeLayers(model, method, n_target_classes):
    if method == "lp":
        # Freeze all layers except the final classification head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.head.parameters():
            param.requires_grad = True

    elif method == "ft":
        # Full fine-tuning: all layers are trainable
        for param in model.parameters():
            param.requires_grad = True

    elif method == "st":
        # Side-network tuning: freeze the main model and add a side network
        for param in model.parameters():
            param.requires_grad = False
        # Example side network (you can customize this)
        model.side_network = nn.Sequential(
            nn.Linear(model.head.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_target_classes)
        )
        for param in model.side_network.parameters():
            param.requires_grad = True


def buildTransform(imnet_norm: bool, imsize: int = 224, extratransforms = None) -> transforms.Compose:
    """
    Builds a transform that modifies input images in a dataset.\n
    imnet_norm: if true, transforms.Normalize images with mean and std of ImageNet.\n
    imsize: required size for input images.\n
    extratransforms gets appended to the returned transforms.Compose object. Note that
    these get applied last! (though for rotation it does not seem to matter that there
    are black patches around).\n
    """
    if extratransforms != None and not isinstance(extratransforms, list):
        extratransforms = [extratransforms]
    
    tfs = [
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize)
    ]

    if imnet_norm:
        tfs += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    
    if extratransforms != None:
        # NOTE: these get applied last, so something like a rotation will show plack around it
        tfs += extratransforms
    
    return transforms.Compose(tfs)