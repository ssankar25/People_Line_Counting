import torch
import torch.nn as nn
import torch.nn.functional as F

# Set whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNnet(torch.nn.Module):
    """
    This class implements a convolutional neural network (CNN) with a series of convolutional and pooling layers 
    followed by dropout and fully connected layers. Designed for binary classification tasks, the network processes 
    single-channel images (e.g., grayscale) and provides logits for two categories.

    Attributes:
        conv1 (nn.Sequential): First convolutional block with convolution, ReLU activation, and max pooling.
        conv2 (nn.Sequential): Second convolutional block with convolution, ReLU activation, and max pooling.
        conv3 (nn.Sequential): Third convolutional block with convolution, ReLU activation, and max pooling.
        conv4 (nn.Sequential): Fourth convolutional block with convolution, ReLU activation, and max pooling.
        conv5 (nn.Sequential): Fifth convolutional block with convolution, ReLU activation, and max pooling.
        fc1 (nn.Linear): Fully connected layer that reduces dimension to 128.
        dropout (nn.Dropout): Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0.
        fc2 (nn.Linear): Final fully connected layer that outputs the logits for the two categories.
    """
    
    def __init__(self):
        """
        Initializes the CNN network with five convolutional layers followed by max pooling and two fully connected layers.
        The network processes input with 1 channel (e.g., grayscale images) and outputs logits for two categories.

        The architecture is as follows:
        - 5 Convolutional Layers each followed by ReLU activation and Max Pooling.
        - A Fully Connected Layer with 128 neurons, followed by ReLU activation and dropout.
        - A Final Fully Connected Layer that outputs the logits for two categories.
        """

        super(CNNnet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),  # Input channels = 1, Output chanels = 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Fourth convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Fifth convolutional layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Determine input size for the fully connected layer based on the feature map size after convolutional layers
        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.dropout = nn.Dropout(p=0.5)
        # Final classification layer, outputting two categories
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        Defines the forward pass of the CNN network.

        Args:
            x (Tensor): The input tensor containing a batch of images.

        Returns:
            x (Tensor): The output tensor containing logits for two categories.
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
