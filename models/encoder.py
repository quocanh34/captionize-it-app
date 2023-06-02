import torch
import torch.nn as nn
import torchvision.models as models
import statistics

class Encoder(nn.Module):
    def __init__(self, embed_size, trainEncoder=False):
        super(Encoder, self).__init__()
        self.trainEncoder = trainEncoder
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features[0]))


