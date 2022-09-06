"""
Dataset lightning class
"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.datamodules.components.kitti_dataset import KITTIDataset, KITTIDataset2, KITTIDataset3

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
        self.KITTI_train = KITTIDataset2(self.hparams.dataset_path, self.hparams.train_sets)
        self.KITTI_val = KITTIDataset2(self.hparams.dataset_path, self.hparams.val_sets)
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

    from time import time

    start1 = time()
    datamodule1 = KITTIDataModule(
        dataset_path='./data/KITTI',
        train_sets='./data/KITTI/train_95.txt',
        val_sets='./data/KITTI/val_95.txt',
        test_sets='./data/KITTI/test_95.txt',
        batch_size=5,
    )
    datamodule1.setup()
    trainloader = datamodule1.val_dataloader()

    for img, label in trainloader:
        print(label["Orientation"])
        break

    results1 = (time() - start1) * 1000

    start2 = time()
    datamodule2 = KITTIDataModule3(
        dataset_path='./data/KITTI',
        train_sets='./data/KITTI/train_95.txt',
        val_sets='./data/KITTI/val_95.txt',
        test_sets='./data/KITTI/test_95.txt',
        batch_size=5,
    )
    datamodule2.setup()
    trainloader = datamodule2.val_dataloader()

    for img, label in trainloader:
        print(label["orientation"])
        break

    results2 = (time() - start2) * 1000

    print(f'Time taken for datamodule1: {results1} ms')
    print(f'Time taken for datamodule2: {results2} ms')
    