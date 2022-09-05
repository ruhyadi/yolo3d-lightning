"""
KITTI Regressor Model
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class RegressorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bins: int,
    ):
        super().__init__()

        # init model
        self.in_features = self._get_in_features(backbone)
        self.model = nn.Sequential(*(list(backbone.children())[:-2]))
        self.bins = bins

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.bins),
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.in_features)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

    def _get_in_features(self, net: nn.Module):

        # TODO: add more models
        in_features = {
            'resnet': (lambda: net.fc.in_features * 7 * 7),
            'vgg': (lambda: net.classifier[0].in_features)
        }
        
        return in_features[(net.__class__.__name__).lower()]()


class RegressorNet2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bins: int,
    ):
        super().__init__()

        # init model
        self.in_features = self._get_in_features(backbone)
        self.model = nn.Sequential(*(list(backbone.children())[:-2]))
        self.bins = bins

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(256, self.bins*2), # 4 bins
            nn.LeakyReLU(0.1)
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(256, self.bins),
            nn.LeakyReLU(0.1)
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, 3), # x, y, z
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.in_features)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

    def _get_in_features(self, net: nn.Module):

        # TODO: add more models
        in_features = {
            'resnet': (lambda: net.fc.in_features * 7 * 7),
            'vgg': (lambda: net.classifier[0].in_features)
        }
        
        return in_features[(net.__class__.__name__).lower()]()


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    """
    Orientation loss function
    """
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()


def orientation_loss2(y_pred, y_true):
    """
    Orientation loss function
    input:  y_true -- (batch_size, bin, 2) ground truth orientation value in cos and sin form.
            y_pred -- (batch_size, bin, 2) estimated orientation value from the ConvNet
    output: loss -- loss values for orientation
    """

    # sin^2 + cons^2
    anchors = torch.sum(y_true ** 2, dim=2)
    # check which bin valid
    anchors = torch.gt(anchors, 0.5)
    # add valid bin
    anchors = torch.sum(anchors.type(torch.float32), dim=1)

    # cos(true)cos(estimate) + sin(true)sin(estimate)
    loss = (y_true[:, : ,0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    # the mean value in each bin
    loss = torch.sum(loss, dim=1) / anchors
    # sum the value at each bin
    loss = torch.sum(loss)
    loss = 2 - 2 * loss

    return loss

def get_model(backbone: str):
    """
    Get truncated model and in_features
    """

    # list of support model name
    # TODO: add more models 
    list_model = ['resnet18', 'vgg11']
    # model_name = str(backbone.__class__.__name__).lower()
    assert backbone in list_model, f"Model not support, please choose {list_model}"

    # TODO: change if else with attributes
    in_features = None
    model = None
    if backbone == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        in_features = backbone.fc.in_features * 7 * 7
        model = nn.Sequential(*(list(backbone.children())[:-2]))
    elif backbone == 'vgg11':
        backbone = models.vgg11(pretrained=True)
        in_features = backbone.classifier[0].in_features
        model = backbone.features

    return [model, in_features]


if __name__ == '__main__':
    
    # from torchvision.models import resnet18
    # from torchsummary import summary

    # backbone = resnet18(pretrained=False)
    # model = RegressorNet(backbone, 2)

    # input_size = (3, 224, 224)
    # summary(model, input_size, device='cpu')

    # test orientation loss
    y_true = torch.tensor([[[0.0, 0.0], [0.9362, 0.3515]]])
    y_pred = torch.tensor([[[0.0, 0.0], [0.9362, 0.3515]]])

    print(y_true, "\n", y_pred)
    print(orientation_loss2(y_pred, y_true))