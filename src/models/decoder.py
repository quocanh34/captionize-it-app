import torch
import torch.nn as nn
import torchvision.models as models
import statistics

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """
        Decoder module for image captioning.
        
        Args:
            embed_size (int): The size of the word embeddings.
            hidden_size (int): The size of the hidden state in the LSTM.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of layers in the LSTM.
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        Forward pass of the decoder model.
        
        Args:
            features (torch.Tensor): Image features of shape (batch_size, embed_size).
            captions (torch.Tensor): Caption sequences of shape (seq_length, batch_size).
        
        Returns:
            torch.Tensor: Predicted scores for each word in the vocabulary, of shape
                (seq_length, batch_size, vocab_size).
        """
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
