import torch
import torch.nn as nn
import torchvision.models as models
import statistics
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder model for image captioning.
    Args:
        embed_size (int): The size of the output feature embeddings from the encoder.
        hidden_size (int): The size of the hidden state in the decoder LSTM.
        vocab_size (int): The size of the vocabulary.
        num_layers (int): The number of layers in the decoder LSTM.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """
        Forward pass of the encoder-decoder model.
        Args:
            images (torch.Tensor): Input image tensors of shape (batch_size, channels, height, width).
            captions (torch.Tensor): Caption sequences of shape (seq_length, batch_size).
        Returns:
            torch.Tensor: Predicted scores for each word in the vocabulary, of shape
                (seq_length, batch_size, vocab_size).
        """
        features = self.encoder.forward(images)
        outputs = self.decoder.forward(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        """
        Generate a caption for an input image.
        Args:
            image (torch.Tensor): Input image tensor of shape (channels, height, width).
            vocabulary (torchtext.vocab.Vocab): Vocabulary object containing word-to-index mapping.
            max_length (int, optional): Maximum length of the generated caption. Default is 50.
        Returns:
            list: List of strings representing the generated caption.
        """
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states =  self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(dim=0)    
                result_caption.append(predicted.item())
                x = self.decoder.embedding(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
