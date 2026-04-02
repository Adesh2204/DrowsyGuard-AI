from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_eye_train_transforms(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eye_val_transforms(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
