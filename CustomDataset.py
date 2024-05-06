import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class that extends PyTorch's Dataset class to handle image loading, transformation, 
    and labeling from a specified directory and annotations file.

    Attributes:
        annotations_file (str): Path to the CSV file containing image file names and their corresponding labels.
        img_dir (str): Directory path where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Initializes the dataset with the directory of images and the annotations file.
        
        Args:
            annotations_file (str): Path to the CSV file containing image file names and labels.
            img_dir (str): Directory path where images are stored.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        """
        self.img_labels = pd.read_csv(annotations_file)  # Load image labels from a CSV file into a Pandas DataFrame.
        self.img_dir = img_dir  # Directory where images are stored.
        self.transform = transform  # Transformations to apply to each image (if any).

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of images (rows in the DataFrame).
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Fetches the image and its label at the specified index, applies transformations, and returns the image and label.

        Args:
            idx (int): Index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label as a tensor.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # Construct the full image path.
        image = Image.open(img_path).convert('L')  # Open and convert the image to grayscale.
        label = self.img_labels.iloc[idx, 1]  # Retrieve the label associated with the image.
        label = torch.tensor(label, dtype=torch.long)  # Convert the label into a PyTorch tensor.

        if self.transform:
            image = self.transform(image)  # Apply the specified transformations.

        return image, label