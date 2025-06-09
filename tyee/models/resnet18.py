import timm
import torch
def resnet18(num_classes, pretrained=True, pretrained_cfg_overlay=None, checkpoint_path=None, **kwargs):
    """
    ResNet18 model for gesture classification.
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, use a pre-trained model.
        
    Returns:
        model: ResNet18 model.
    """
    # Load the ResNet18 model with the specified number of classes
    model = timm.create_model('resnet18', pretrained=pretrained, pretrained_cfg_overlay=pretrained_cfg_overlay, num_classes=num_classes)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            # print(checkpoint.keys())
            print(f"Loaded model weights from {checkpoint_path} without 'model' key")
            # print(f"No 'model' key found in checkpoint: {checkpoint_path}")
    return model