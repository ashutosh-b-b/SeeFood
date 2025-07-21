from bounding_box import multibox_detection, multibox_prior
import torch 
from torch import nn 
from torch.nn import functional as F

class ClassPredictionLayer(nn.Module):
    def __init__(self, num_inputs, num_anchors, num_classes):
        super().__init__()
        self.net = nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
    def forward(self, x):
        return self.net(x)

class BBoxPredictionLayer(nn.Module):
    def __init__(self, num_inputs, num_anchors):
        super().__init__()
        self.net = nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
    def forward(self, x):
        return self.net(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.net(x)

class BaseNet(nn.Module):
    def __init__(self, channels = [16, 32, 64], image_channels = 3):
        super().__init__()
        blocks = [] 
        channels = [image_channels] + channels
        for i in range(len(channels) - 1):
            blocks.append(
                VGGBlock(channels[i], channels[i+1])
            )
        self.net = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.net(x)
        
class MultiScaleFeatureLayer(nn.Module): 
    def __init__(self, block, cls_pred_layer, bbox_pred_layer, size, ratio):
        super().__init__()
        self.block = block 
        self.cls_predictor = cls_pred_layer
        self.bbox_predictor = bbox_pred_layer
        self.size = size 
        self.ratio = ratio 
    def forward(self, x):
        Y = self.block(x)
        anchors = multibox_prior(Y, sizes=self.size, ratios=self.ratio)
        cls_preds = self.cls_predictor(Y)
        bbox_preds = self.bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)
        
class SSD(nn.Module): 
    def __init__(self, num_classes, 
                 base_channels = [16, 32, 64], 
                 intermediate_channel = 128, 
                 num_blocks=4, 
                 sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                 ratios = [1, 2, 0.5],
                 **kwargs):
        
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.base_channels = base_channels 
        self.intermediate_channel = intermediate_channel
        ratios = [ratios]*len(sizes)
        self.num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        self.sizes = sizes 
        self.ratios = ratios
        self.num_blocks = num_blocks
        base_layer = MultiScaleFeatureLayer(BaseNet(channels = base_channels),
            ClassPredictionLayer(base_channels[-1], self.num_anchors, num_classes), 
            BBoxPredictionLayer(base_channels[-1], self.num_anchors),
            sizes[0],
            ratios[0],
        )
        setattr(self, "layer_0", base_layer)
        curr_channel = base_channels[-1]
        for i in range(num_blocks):
            if i == num_blocks - 1:
                block = nn.AdaptiveMaxPool2d((1,1))
            else:
                block = VGGBlock(curr_channel, intermediate_channel)
            layer = MultiScaleFeatureLayer(block,
                ClassPredictionLayer(intermediate_channel, self.num_anchors, num_classes), 
                BBoxPredictionLayer(intermediate_channel, self.num_anchors),
                sizes[i],
                ratios[i],
            )
            setattr(self, f'layer_{i+1}', layer)
            curr_channel = intermediate_channel
    
    def flatten_pred(self, pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    def concat_preds(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1) 
    
    def forward(self, x): 
        total_blocks = self.num_blocks + 1
        
        anchors, cls_preds, bbox_preds = [None] *total_blocks , [None] * total_blocks, [None] * total_blocks
        for i in range(total_blocks):
            layer = getattr(self, f'layer_{i}')
            x, anchors[i], cls_preds[i], bbox_preds[i] = layer(x) 
        anchors = torch.cat(anchors, dim=1)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
        
def predict(net, X, device):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

