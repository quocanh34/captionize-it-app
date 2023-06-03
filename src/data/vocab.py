import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from nltk import word_tokenize
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class Vocabulary():
    """
    Vocabulary class for handling word-to-index and index-to-word mappings.

    Args:
        freq_threshold (int): Frequency threshold for including words in the vocabulary.
    """

    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        return [token.lower() for token in word_tokenize(text)]
    
    def build_vocabulary(self, sentence_list):
        """
        Builds the vocabulary from a list of sentences.

        Args:
            sentence_list (list): List of sentences.

        """
        frequecies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequecies:
                    frequecies[word] = 1

                else:
                    frequecies[word] += 1

                if frequecies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        """
        Converts a text into a list of numericalized tokens.
        Args:
            text (str): Input text.
        Returns:
            list: List of numericalized tokens.
        """
        tokenized_text = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]




