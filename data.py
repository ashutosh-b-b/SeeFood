import os
import json
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as TF

class SeeFoodCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transforms=None, size=512):
        self.root = root
        self.transforms = transforms 
        self.to_tensor = ToTensor()
        self.size = (size, size)
        with open(ann_file) as f:
            self.coco_data = json.load(f)

        self.image_id_to_info = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(self.image_id_to_info.keys())
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        self.idx_to_labels = {v: k for k, v in self.cat_id_to_name.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_id_to_info[img_id]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = TF.resize(image, self.size)
        image = self.to_tensor(image)

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        new_w, new_h = self.size
        boxes = torch.tensor(boxes[0], dtype=torch.float32).reshape(1, -1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (1 / orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (1 / orig_h)
        labels = torch.tensor([labels[0]], dtype=torch.int64).reshape(-1, 1)
        target = torch.cat((labels, boxes), dim =1 )
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
