"""
KITTI Regressor Model
"""
import torch
from torch import nn
from pytorch_lightning import LightningModule

from src.models.components.base import OrientationLoss, orientation_loss2


class RegressorModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.0001,
        momentum: float = 0.9,
        w: float = 0.4,
        alpha: float = 0.6,
    ):
        super().__init__()

        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # init model
        self.net = net

        # loss functions
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = OrientationLoss

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch):
        x, y = batch

        # convert to float
        x = x.float()
        truth_orient = y["Orientation"].float()
        truth_conf = y["Confidence"].float()
        truth_dim = y["Dimensions"].float()

        # predict y_hat
        preds = self(x)
        [orient, conf, dim] = preds

        # compute loss
        orient_loss = self.orient_loss_func(orient, truth_orient, truth_conf)
        dim_loss = self.dim_loss_func(dim, truth_dim)
        truth_conf = torch.max(truth_conf, dim=1)[1]
        conf_loss = self.conf_loss_func(conf, truth_conf)
        loss_theta = conf_loss + self.hparams.w * orient_loss
        loss = self.hparams.alpha * dim_loss + loss_theta

        return [loss, loss_theta, orient_loss, dim_loss, conf_loss], preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "train/loss": loss[0],
                "train/theta_loss": loss[1],
                "train/orient_loss": loss[2],
                "train/dim_loss": loss[3],
                "train/conf_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "val/loss": loss[0],
                "val/theta_loss": loss[1],
                "val/orient_loss": loss[2],
                "val/dim_loss": loss[3],
                "val/conf_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()

        # log to tensorboard
        self.log("val/avg_loss", avg_val_loss)
        return {"loss": avg_val_loss}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
        #     momentum=self.hparams.momentum
        # )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        return optimizer

class RegressorModel2(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.0001,
        momentum: float = 0.9,
        w: float = 0.4,
        alpha: float = 0.6,
    ):
        super().__init__()

        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # init model
        self.net = net

        # loss functions
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = orientation_loss2

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch):
        x, y = batch

        # convert to float
        x = x.float()
        gt_orient = y["orientation"].float()
        gt_conf = y["confidence"].float()
        gt_dims = y["dimensions"].float()

        # predict y_true
        predictions = self(x)
        [pred_orient, pred_conf, pred_dims] = predictions

        # compute loss
        loss_orient = self.orient_loss_func(pred_orient, gt_orient)
        loss_dims = self.dim_loss_func(pred_dims, gt_dims)
        gt_conf = torch.max(gt_conf, dim=1)[1]
        loss_conf = self.conf_loss_func(pred_conf, gt_conf)
        # weighting loss => see paper
        loss_theta = loss_conf + (self.hparams.w * loss_orient)
        loss = (self.hparams.alpha * loss_dims) + loss_theta

        return [loss, loss_theta, loss_orient, loss_conf, loss_dims], predictions, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "train/loss": loss[0],
                "train/theta_loss": loss[1],
                "train/orient_loss": loss[2],
                "train/conf_loss": loss[3],
                "train/dim_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "val/loss": loss[0],
                "val/theta_loss": loss[1],
                "val/orient_loss": loss[2],
                "val/conf_loss": loss[3],
                "val/dim_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()

        # log to tensorboard
        self.log("val/avg_loss", avg_val_loss)
        return {"loss": avg_val_loss}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

        return optimizer

class RegressorModel3(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.0001,
        momentum: float = 0.9,
        w: float = 0.4,
        alpha: float = 0.6,
    ):
        super().__init__()

        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # init model
        self.net = net

        # loss functions
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = OrientationLoss

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch):
        x, y = batch

        # convert to float
        x = x.float()
        gt_orient = y["orientation"].float()
        gt_conf = y["confidence"].float()
        gt_dims = y["dimensions"].float()

        # predict y_true
        predictions = self(x)
        [pred_orient, pred_conf, pred_dims] = predictions

        # compute loss
        loss_orient = self.orient_loss_func(pred_orient, gt_orient, gt_conf)
        loss_dims = self.dim_loss_func(pred_dims, gt_dims)
        gt_conf = torch.max(gt_conf, dim=1)[1]
        loss_conf = self.conf_loss_func(pred_conf, gt_conf)
        # weighting loss => see paper
        loss_theta = loss_conf + (self.hparams.w * loss_orient)
        loss = (self.hparams.alpha * loss_dims) + loss_theta

        return [loss, loss_theta, loss_orient, loss_conf, loss_dims], predictions, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "train/loss": loss[0],
                "train/theta_loss": loss[1],
                "train/orient_loss": loss[2],
                "train/conf_loss": loss[3],
                "train/dim_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "val/loss": loss[0],
                "val/theta_loss": loss[1],
                "val/orient_loss": loss[2],
                "val/conf_loss": loss[3],
                "val/dim_loss": loss[4],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss[0], "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()

        # log to tensorboard
        self.log("val/avg_loss", avg_val_loss)
        return {"loss": avg_val_loss}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        return optimizer

if __name__ == "__main__":

    from src.models.components.base import RegressorNet
    from torchvision.models import resnet18

    model1 = RegressorModel(
        net=RegressorNet(backbone=resnet18(pretrained=False), bins=2),
    )

    print(model1)

    model2 = RegressorModel3(
        net=RegressorNet(backbone=resnet18(pretrained=False), bins=2),
    )

    print(model2)