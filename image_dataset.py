# Description: Image dataset class.
#
# Input:
#   - files_path: path to images.
#   - width: width of the image.
#   - height: height of the image.
#   - classes: list of classes.
#   - transforms: data augmentation.

# Libraries.
import os # Operating system library.
import cv2 # OpenCV library.
import numpy as np # Numerical python library.
import torch # PyTorch library.
import xml.etree.ElementTree as et # XML library.

# Image dataset class.
class ImageDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    files_path,
    width,
    height,
    classes = None,
    transforms=None):
      self.files_path = files_path # Path to images.
      self.width = width # Width of the image.
      self.height = height # Height of the image.
      self.transforms = transforms # Data augmentation.
      self.image_paths = [os.path.join(files_path, f) for f in os.listdir(files_path) if f.endswith('.jpg')] # List of images.
      self.classes = classes # List of classes.
      self.num_classes = len(self.classes) # Number of classes.

  # Length of the dataset.
  def __len__(self):
    return len(self.image_paths)

  # Get item.
  def __getitem__(self, index):
      # Load image.
      img_path = self.image_paths[index]
      img = cv2.imread(img_path)

      # Get shape.
      h, w, _ = img.shape

      # BGR to RGB and data type changed.
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

      # Resize image.
      img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
      img /= 255.0

      # Load annotation file.
      annot_path = img_path.replace('.jpg', '.xml')
      tree = et.parse(annot_path)
      root = tree.getroot()

      # Extract bounding box coordinates and labels.
      boxes, labels = [], []
      for obj in root.findall('object'):
        # Label of the object.
        label = obj.find('name').text.lower().strip()

        # If the label is not in the list of classes, skip it.
        if label not in self.classes:
          continue
        # Bounding box.
        bbox = obj.find('bndbox')

        # Cords.
        x_min = ( float(bbox.find('xmin').text) / w ) * self.width
        x_max = ( float(bbox.find('xmax').text) / w ) * self.width
        y_min = ( float(bbox.find('ymin').text) / h ) * self.height
        y_max = ( float(bbox.find('ymax').text) / h ) * self.height

        # Append bounding box and label.
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(self.classes.index(label))

      # Convert boxes and labels to tensors.
      if len(labels) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
      else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

      # Compute area of boxes and is_crowd.
      area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
      iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

      # Define target dictionary.
      target = {
        'boxes': boxes,
        'labels': labels,
        'image_id': torch.tensor([index]),
        'area': area,
        'iscrowd': iscrowd
      }

      # Apply data augmentation if specified.
      if self.transforms:
        # Transform image and target.
        transformed = self.transforms(image=img,
                                      bboxes=target['boxes'],
                                      labels=target['labels'])
        # Get transformed image and target.
        img = transformed['image']

        # Get transformed bounding boxes.
        target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)

      return img, target