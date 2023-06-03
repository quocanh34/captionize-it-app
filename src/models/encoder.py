import torch
import torch.nn as nn
import torchvision.models as models
import statistics

class Encoder(nn.Module):
    def __init__(self, embed_size, trainEncoder=False):
        """
        Encoder module for image captioning.
        
        Args:
            embed_size (int): The size of the output feature embeddings.
            trainEncoder (bool, optional): Flag indicating whether to train the encoder's parameters or not.
                Default is False.
        """
        super(Encoder, self).__init__()
        self.trainEncoder = trainEncoder
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        """
        Forward pass of the encoder model.
        
        Args:
            images (torch.Tensor): Input image tensors of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output feature embeddings of shape (batch_size, embed_size).
        """
        features = self.inception(images)
        return self.dropout(self.relu(features[0]))


