from bounding_box import multibox_target, display
from model import SSD, predict, per_class_accuracy
from loss import calc_loss, cls_eval, bbox_eval
from data import SeeFoodCocoDataset 
import mlflow 
import json
import statistics
from statistics import mean
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Subset, DataLoader
import torch
import argparse
from pred_score_idxs import IDXS

def load_dataset(config):
    train_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/train",
        ann_file="SeeFoodDataset/train/_annotations.coco.json"
    )
    val_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/valid",
        ann_file="SeeFoodDataset/valid/_annotations.coco.json"
    )
    test_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/test",
        ann_file="SeeFoodDataset/test/_annotations.coco.json"
    )
    batch_size = config["batch_size"]

    train_iter = torch.utils.data.DataLoader(train_dataset, config["batch_size"], shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, config["batch_size"])
    test_iter = torch.utils.data.DataLoader(test_dataset, config["batch_size"])
    return train_iter, val_iter, test_iter

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

def get_device(device):
    return torch.device(f'cuda:{device}')

def train_model(net, data, config): 
    device = get_device(config["device"])

    train_iter, val_iter = data 

    num_epochs = config["num_epochs"]
    lr = config["lr"]

    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    
    net = net.to(device)
    
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        net.train()
        losses = {"train_loss" : [], 
                "train_cls_error": [], 
                "train_bbox_error": [], 
                "val_loss": [],
                "val_cls_error": [],
                "val_bbox_error": [],
                }
        
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)

            anchors, cls_preds, bbox_preds = net(X)

            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
    
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
            l.mean().backward()
            trainer.step()
            with torch.no_grad():
                cls_error = cls_eval(cls_preds, cls_labels) / cls_labels.numel()
                bbox_error = bbox_eval(bbox_preds, bbox_labels, bbox_masks) / bbox_labels.numel()
            
                train_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
                train_cls_err, train_bbox_mae = 1 - cls_error, bbox_error
            
                losses["train_loss"].append(train_loss.mean().tolist())
                losses["train_cls_error"].append(train_cls_err)
                losses["train_bbox_error"].append(train_bbox_mae)
        with torch.no_grad():
            for features, target in val_iter:
                X, Y = features.to(device), target.to(device)
                anchors, cls_preds, bbox_preds = net(X)
        
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        
                val_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                            bbox_masks)
                cls_error = cls_eval(cls_preds, cls_labels) / cls_labels.numel()
                bbox_error = bbox_eval(bbox_preds, bbox_labels, bbox_masks) / bbox_labels.numel()
                
                val_cls_err, val_bbox_mae = 1 - cls_error, bbox_error
                losses["val_loss"].append(val_loss.mean().tolist())
                losses["val_cls_error"].append(val_cls_err)
                losses["val_bbox_error"].append(val_bbox_mae)

        mean_train_loss = mean(losses["train_loss"])
        mean_val_loss = mean(losses["val_loss"])
        mean_train_class_err = mean(losses["train_cls_error"])
        mean_val_class_err = mean(losses["val_cls_error"])
        mean_train_bbox_err = mean(losses["train_bbox_error"])
        mean_val_bbox_err = mean(losses["val_bbox_error"])

        mlflow.log_metrics(
            {
                "train_loss": mean_train_loss,
                "train_cls_error": mean_train_class_err,
                "train_bbox_error": mean_train_bbox_err,
                "val_loss": mean_val_loss,
                "val_cls_error": mean_val_class_err,
                "val_bbox_error": mean_val_bbox_err
                
            },
            step=epoch+1,
        )
        print(f'Epoch {epoch+1}, Train Loss: {mean_train_loss}, Val Loss: {mean_val_loss}, Train ClsErr: {mean_train_class_err}, Train Bbox Err: {mean_train_bbox_err}, Val ClsErr: {mean_val_class_err}, Val Bbox Err: {mean_val_bbox_err}')
    
def test_model(net, test_iter, device):
    test_losses = []
    test_cls_errors = []
    test_bbox_errors = []

    with torch.no_grad():
        for features, target in test_iter:
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
    
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
    
            loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
            cls_error = cls_eval(cls_preds, cls_labels) / cls_labels.numel()
            bbox_error = bbox_eval(bbox_preds, bbox_labels, bbox_masks) / bbox_labels.numel()
            
            test_cls_err, test_bbox_mae = 1 - cls_error, bbox_error
            test_losses.append(loss.mean().tolist())
            test_cls_errors.append(test_cls_err)
            test_bbox_errors.append(test_bbox_mae)
        mlflow.log_metrics(
            {
                "test_loss": mean(test_losses),
                "test_cls_error": mean(test_cls_errors),
                "test_bbox_error": mean(test_bbox_errors)                
            },
        )

def save_random_predictions(model, test_idxs, device, run_id, threshold=0.03):    
    with torch.no_grad():
        test_dataset = SeeFoodCocoDataset(
            root="SeeFoodDataset/test",
            ann_file="SeeFoodDataset/test/_annotations.coco.json"
        )
        for idx in test_idxs:
            x = test_dataset[idx][0]
            img = to_pil_image(x)
            x = x.to(device)
            output = predict(model, x.unsqueeze(0))
            fig = display(img, output.cpu(), threshold, test_dataset.cat_id_to_name)
            filename = f"artifacts/display_{run_id}_{idx}.png"
            fig.savefig(filename)

            # Log to MLflow
            mlflow.log_artifact(filename)
        
def log_prediction_score(model, idxs, device):
    train_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/train",
        ann_file="SeeFoodDataset/train/_annotations.coco.json"
    )
    val_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/valid",
        ann_file="SeeFoodDataset/valid/_annotations.coco.json"
    )
    test_dataset = SeeFoodCocoDataset(
        root="SeeFoodDataset/test",
        ann_file="SeeFoodDataset/test/_annotations.coco.json"
    )
    with torch.no_grad():
        for (dataset, type) in zip((train_dataset, val_dataset, test_dataset), ("train", "val", "test")):
            conf = 0
            comm = 0
            for i in idxs:
                feature, batch = dataset[i]
                x, y = feature.to(device), batch.to(device)
                out = predict(model, x.unsqueeze(0))
                if out.shape[0] <= 1:
                    continue
                classes = out[0, :].int()
                high_confidence_prediction = classes[torch.argmax(out[1, :])]
                vals, counts = torch.unique(classes, return_counts=True)
                most_common_prediction = vals[torch.argmax(counts)]
                conf += high_confidence_prediction.item() == int(y[0][0])
                comm += most_common_prediction.item() == int(y[0][0])

            mlflow.log_metrics(
                {
                    f'{type}_confidence_score' : conf / len(idxs),
                    f'{type}_common_score' : comm / len(idxs),
                }
            )

def log_per_class_accuracy(model, device, train_iter, val_iter, test_iter, run_id):
    train_p_acc = per_class_accuracy(model, train_iter, device)
    val_p_acc = per_class_accuracy(model, val_iter, device)
    test_p_acc = per_class_accuracy(model, test_iter, device)
    train_acc_path = f"artifacts/train_per_class_acc_{run_id}.json"
    val_acc_path = f"artifacts/val_per_class_acc_{run_id}.json"
    test_acc_path = f"artifacts/test_per_class_acc_{run_id}.json"

    with open(train_acc_path, "w") as f:
        json.dump(train_p_acc, f)

    # mlflow.log_artifact(train_acc_path)

    with open(val_acc_path, "w") as f:
        json.dump(val_acc_path, f)

    # mlflow.log_artifact(val_acc_path)
    
    with open(test_acc_path, "w") as f:
        json.dump(test_acc_path, f)

    # mlflow.log_artifact(test_acc_path)

def main(id):
    with open('actual_params_3.json', 'r') as f:
        config = json.load(f)[id]
    data_config = config["data_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    
    train_iter, val_iter, test_iter = load_dataset(data_config)
    model = configure_model(model_config)
    mlflow.set_tracking_uri("http://127.0.0.1:8081")
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run() as run:
         mlflow.set_tag("run_tag", config["tag"])
         mlflow.log_params(data_config)
         mlflow.log_params(model_config)
         mlflow.log_params(train_config)
         train_model(model, (train_iter, val_iter), train_config)

         full_model_path = f"artifacts/model_final_full_{run.info.run_id}.pth"
         torch.save(model, full_model_path)
                  
         state_model_path = f"artifacts/model_final_state_{run.info.run_id}.pth"
         torch.save(model.state_dict(), state_model_path)

         mlflow.log_artifact(full_model_path)
         mlflow.log_artifact(state_model_path)


         device = get_device(train_config["device"])
         test_model(model, test_iter, device)
         log_prediction_score(model, IDXS, device)
         log_per_class_accuracy(model, device, train_iter, val_iter, test_iter, run.info.run_id)
         save_random_predictions(model, config["test_idxs"], device, run.info.run_id, threshold=0.03)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model based on config ID")
    parser.add_argument("--id", type=int, required=True, help="Index in params.json list")
    args = parser.parse_args()

    main(args.id)
        
    




    