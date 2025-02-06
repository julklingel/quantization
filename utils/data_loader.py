
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
from torchvision import datasets, transforms


def load_data_classifier(data_dir, batch_size, shuffle=True, resize_x=224, resize_y=224):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_img_dir = os.path.join(data_dir, 'train2017', 'all')
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_img_dir = os.path.join(data_dir, 'val2017', 'all')
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    train_dataset = CocoClassificationDataset(train_img_dir, train_ann_file, transforms=transform)
    val_dataset = CocoClassificationDataset(val_img_dir, val_ann_file, transforms=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn_classifier)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn_classifier)

    class_names = {0: 'cat', 1: 'dog'}

    return trainloader, valloader, class_names


class DetectionTransform:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        orig_width, orig_height = image.size
        new_width, new_height = self.size

        resized_image = F.resize(image, (new_height, new_width))

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        if target and 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            scaled_boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=torch.float32)
            target['boxes'] = scaled_boxes

        resized_image = F.to_tensor(resized_image)
        resized_image = F.normalize(resized_image, mean=self.mean, std=self.std)

        return resized_image, target


class CocoClassificationDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms
        self.cat_ids = self.coco.getCatIds(catNms=['cat', 'dog'])
        self.valid_ids = self._filter_valid_ids()

    def _filter_valid_ids(self):
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            if ann_ids:
                valid_ids.append(img_id)
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, index):
        img_id = self.valid_ids[index]

        try:
            image = self._load_image(img_id)
            anns = self._load_target(img_id)
        except Exception as e:
            print(f"Error loading image {img_id}: {str(e)}")
            return None, None

        labels = [ann['category_id'] for ann in anns]

        # Map category IDs to 0 (cat) and 1 (dog)
        labels = [0 if label == self.coco.getCatIds(catNms=['cat'])[0] else 1 for label in labels]

        # Ensure only one label per image
        label = labels[0] if labels else -1

        target = {
            'labels': torch.tensor(label, dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target['labels']
    
    
def collate_fn_classifier(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    images, targets = zip(*batch)
    
    # Ensure targets are 1D tensors
    targets = [torch.tensor([t]) if t.dim() == 0 else t for t in targets]
    
    # Flatten the targets to a 1D tensor
    targets = torch.cat(targets)
    
    return torch.stack(images), targets


def load_data_detection(data_dir, batch_size, shuffle=True):
    transform = DetectionTransform(
        size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_img_dir = os.path.join(data_dir, 'train2017', "all")
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_img_dir = os.path.join(data_dir, 'val2017', "all")
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    train_dataset = CatCocoDataset(train_img_dir, train_ann_file, transforms=transform)
    val_dataset = CatCocoDataset(val_img_dir, val_ann_file, transforms=transform)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )

    return trainloader, valloader


class CatCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms
        self.cat_ids = self.coco.getCatIds(catNms=['cat'])
        if not self.cat_ids:
            raise ValueError("'cat' category not found in annotations")
        self.cat_id = self.cat_ids[0]
        self.valid_ids = self._filter_valid_ids()

    def _filter_valid_ids(self):
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_id)
            if ann_ids:  
                valid_ids.append(img_id)
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, index):
        img_id = self.valid_ids[index]
        
        try:
            image = self._load_image(img_id)
            anns = self._load_target(img_id)
        except Exception as e:
            print(f"Error loading image {img_id}: {str(e)}")
            return None, None


        boxes = []
        for ann in anns:
            if ann['category_id'] == self.cat_id:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.ones((len(boxes),), dtype=torch.int64) if boxes else torch.zeros((0,), dtype=torch.int64)
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))  
    return tuple(zip(*batch))

def load_data_detection(data_dir, batch_size, shuffle=True):
    transform = DetectionTransform(
        size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


    train_img_dir = os.path.join(data_dir, 'train2017', "all")  
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_img_dir = os.path.join(data_dir, 'val2017', "all")  
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')


    train_dataset = CatCocoDataset(train_img_dir, train_ann_file, transforms=transform)
    val_dataset = CatCocoDataset(val_img_dir, val_ann_file, transforms=transform)


    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )

    return trainloader, valloader