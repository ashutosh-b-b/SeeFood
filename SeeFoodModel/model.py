from bounding_box import multibox_detection, multibox_prior
import torch 
from torch import nn 
from torch.nn import functional as F
import torchvision.models as models
from collections import defaultdict


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

class BaseNetPreTrainedVGG(nn.Module):
    def __init__(self, channels = [64, 128]):
        super().__init__()
        blocks = [] 
        vgg16 = models.vgg16(pretrained=True)
        base_layers = nn.Sequential(*list(vgg16.features.children())[:5])  # conv1 + ReLU + conv2 + ReLU + MaxPool
        for param in base_layers.parameters():
            param.requires_grad = False
        
        # base layer output is 64 channels
        assert channels[0] == 64, "Starting channel for base_channels should be 64"
        blocks.append(base_layers)
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
                 use_pretrained_vgg = False,
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
        if use_pretrained_vgg:
            base_net = BaseNetPreTrainedVGG(channels = base_channels)
        else:
            base_net = BaseNet(channels = base_channels)
        base_layer = MultiScaleFeatureLayer(base_net,
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
        
def predict(net, X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def predict_batch(model, X, device):
    model.eval()
    anchors, cls_preds, bbox_preds = model(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    valid_mask = output[:, :, 0] != -1
    valid_counts = valid_mask.sum(dim=1)
    filtered_output = [img_out[mask] for img_out, mask in zip(output, valid_mask)]
    return filtered_output


def per_class_accuracy(model, data, device):
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            out = predict_batch(model, x, device)
            high_confidence_predictions = []

            for o in out:
                if o.shape[0] < 2 or o.shape[1] == 0:
                    continue  # skip invalid or empty predictions
                classes = o[0, :].int()
                confidences = o[1, :]

                high_conf_idx = torch.argmax(confidences)
                high_conf_pred = classes[high_conf_idx]
                high_confidence_predictions.append(high_conf_pred)

            y_true = y[:, 0, 0].cpu()
            y_pred = torch.stack(high_confidence_predictions).cpu()

            for true, pred in zip(y_true, y_pred):
                true = int(true.item())
                total_per_class[true] += 1
                if pred.item() == true:
                    correct_per_class[true] += 1

    # Final per-class accuracy
    per_class_accuracy = {
        cls: correct_per_class[cls] / total_per_class[cls]
        for cls in total_per_class
    }

    return per_class_accuracy

def configure_model(params): 
    sizes = params["sizes"]
    num_blocks = params["num_blocks"]
    
    assert len(sizes) == num_blocks + 1, "Sizes and Number of blocks donot match"

    base_channels = params["base_channels"]
    intermediate_channel = params["intermediate_channel"]
    num_classes = params["num_classes"]
    sizes = params["sizes"]
    ratios = params["ratios"]
    num_blocks = params["num_blocks"]
    use_pretrained_vgg = params["use_pretrained"]
    model = SSD(
        num_classes, 
        base_channels = base_channels,
        intermediate_channel = intermediate_channel,
        sizes = sizes,
        ratios = ratios,
        num_blocks = num_blocks,
        use_pretrained_vgg = use_pretrained_vgg
    )

    return model 
