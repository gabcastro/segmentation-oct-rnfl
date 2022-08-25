import albumentations as A

class Augmentation:
    def __init__(self) -> None:
        pass
    
    def round_clip_0_1(self, x, **kwargs):
        return x.round().clip(0, 1)

    def get_training_augmentation(self):
        """Define heavy augmentations"""
        train_transform = [

            A.HorizontalFlip(p=0.5),

            # A.ShiftScaleRotate(
            #     scale_limit=0.5,
            #     rotate_limit=0,
            #     shift_limit=0.1,
            #     p=1,
            #     border_mode=0
            # ),

            # A.PadIfNeeded(
            #     min_height=512,
            #     min_width=512,
            #     always_apply=True,
            #     border_mode=0
            # ),

            # A.RandomCrop(
            #     height=512,
            #     width=512,
            #     always_apply=True
            # ),

            # A.GaussNoise(p=0.2),
            # A.Perspective(p=0.5),

            # A.OneOf(
            #     [
            #         A.CLAHE(p=1),
            #         A.RandomBrightnessContrast(p=1),
            #         A.RandomGamma(p=1)
            #     ],
            #     p=0.9
            # ),

            # A.OneOf(
            #     [
            #         A.Sharpen(p=1),
            #         A.Blur(blur_limit=3, p=1),
            #         A.MotionBlur(blur_limit=3, p=1)
            #     ],
            #     p=0.9
            # ),

            # A.OneOf(
            #     [
            #         A.RandomBrightnessContrast(p=1),
            #         A.HueSaturationValue(p=1)
            #     ],
            #     p=0.9
            # ),

            # A.Lambda(mask=self.round_clip_0_1)
        ]

        return A.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.PadIfNeeded(544, 544)
        ]
        return A.Compose(test_transform)

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform
    
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        
        """

        _transform = [
            A.Lambda(image=preprocessing_fn)
        ]
        return A.Compose(_transform)