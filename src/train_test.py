from data.datamodules import ImageNetDataModule, StylizedImageNetDataModule, OODStylizedImageNetDataModule

imagenet_module = ImageNetDataModule()
stylized_imagenet_module = StylizedImageNetDataModule()
ood_stylized_imagenet_module = OODStylizedImageNetDataModule()

imagenet_module.setup()
stylized_imagenet_module.setup()
ood_stylized_imagenet_module.setup()