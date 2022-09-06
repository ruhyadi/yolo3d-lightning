"""
Dataset lightning class
"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.datamodules.components.kitti_dataset import KITTIDataset, KITTIDataset3

class KITTIDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = './data/KITTI',
        train_sets: str = './data/KITTI/train.txt',
        val_sets: str = './data/KITTI/val.txt',
        test_sets: str = './data/KITTI/test.txt',
        batch_size: int = 32,
        num_worker: int = 4,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

        # transforms
        # TODO: using albumentations
        self.dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        """ Split dataset to training and validation """
        self.KITTI_train = KITTIDataset(self.hparams.dataset_path, self.hparams.train_sets)
        self.KITTI_val = KITTIDataset(self.hparams.dataset_path, self.hparams.val_sets)
        # self.KITTI_test = KITTIDataset(self.hparams.dataset_path, self.hparams.test_sets)
        # TODO: add test datasets dan test sets

    def train_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=False
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.KITTI_test,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_worker,
    #         shuffle=False
    #     )

class KITTIDataModule2(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = './data/KITTI',
        train_sets: str = './data/KITTI/train.txt',
        val_sets: str = './data/KITTI/val.txt',
        test_sets: str = './data/KITTI/test.txt',
        batch_size: int = 32,
        num_worker: int = 4,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None):
        """ Split dataset to training and validation """
        self.KITTI_train = KITTIDataLoader(self.hparams.dataset_path, self.hparams.train_sets)
        self.KITTI_val = KITTIDataLoader(self.hparams.dataset_path, self.hparams.val_sets)
        # self.KITTI_test = KITTIDataset(self.hparams.dataset_path, self.hparams.test_sets)
        # TODO: add test datasets dan test sets

    def train_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=False
        )

class KITTIDataModule3(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = './data/KITTI',
        train_sets: str = './data/KITTI/train.txt',
        val_sets: str = './data/KITTI/val.txt',
        test_sets: str = './data/KITTI/test.txt',
        batch_size: int = 32,
        num_worker: int = 4,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

        # transforms
        # TODO: using albumentations
        self.dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        """ Split dataset to training and validation """
        self.KITTI_train = KITTIDataset3(self.hparams.dataset_path, self.hparams.train_sets)
        self.KITTI_val = KITTIDataset3(self.hparams.dataset_path, self.hparams.val_sets)
        # self.KITTI_test = KITTIDataset(self.hparams.dataset_path, self.hparams.test_sets)
        # TODO: add test datasets dan test sets

    def train_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            shuffle=False
        )


if __name__ == '__main__':

    # dataset = KITTIDataModule(batch_size=1)
    # dataset.setup()
    # train = dataset.train_dataloader()

    # for img, label in train:
    #     print(label)
    #     break

    dataset = KITTIDataModule3(batch_size=1)
    dataset.setup()
    train = dataset.train_dataloader()

    for img, label in train:
        print(img.shape)
        print(label)
        break

    # output
    # torch.Size([1, 3, 224, 224])
    # {'orientation': tensor([[[0.9992, 0.0392],
    #          [0.0000, 0.0000]]], dtype=torch.float64), 
    #          'confidence': tensor([[1., 0.]], dtype=torch.float64), 
    #          'dimensions': tensor([[-14.0929, -15.1018, -35.4715]], dtype=torch.float64)}
    