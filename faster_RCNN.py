# Description: Faster R-CNN model.
#
# Input:
#   - num_classes: number of classes.
#
# Output:
#   - predictions: predictions of the model.
#

# Libraries.
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Faster R-CNN model.
from torchvision.models.detection import fasterrcnn_resnet50_fpn # Faster R-CNN model.
import torch.nn as nn # Neural network library.

# Faster R-CNN model class using torchvision backbone.
class FasterRCNNModel(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super(FasterRCNNModel, self).__init__()

        # Load pre-trained model.
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        # Change number of classes.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        # Get predictions.
        predictions = self.model(images, targets)

        # Return predictions.
        return predictions