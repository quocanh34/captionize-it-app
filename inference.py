import torch
import torchvision.transforms as transforms
# import json
from src.models.model import EncoderDecoder
from PIL import Image


def inference(example_path, checkpoint_path, device, vocab_path):
    """
    Generate a caption for the input image.

    Args:
        example_path (str): Path to the input image file.
        checkpoint_path (str): Path to the trained model checkpoint file.
        device (str): Device on which to perform inference ('cuda' or 'cpu').
        vocab_path (str): Path to the vocabulary JSON file.

    Returns:
        str: The generated caption for the input image.
    """

    # Make the transform 
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load the vocabulary from JSON
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Default params
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1

    # Load trained model
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    # Set to eval mode
    model.eval()

    # Get the caption
    test_img1 = transform(Image.open(example_path).convert("RGB")).unsqueeze(0).to(device)
    final_output = " ".join(caption_image(test_img1, vocab, model.encoder, model.decoder))
    final_output = final_output.replace("<SOS>", "")
    final_output = final_output.replace("<EOS>", "")
    final_output = final_output.replace(".", "")
    return final_output

def caption_image(image, vocabulary, encoder, decoder, max_length=50):
    """
    Generate a caption given an image and the trained encoder and decoder models.

    Args:
        image (torch.Tensor): Input image tensor.
        vocabulary (dict): Vocabulary dictionary mapping indices to words.
        encoder (torch.nn.Module): Trained encoder model.
        decoder (torch.nn.Module): Trained decoder model.
        max_length (int, optional): Maximum length of the generated caption. Defaults to 50.

    Returns:
        list: List of words representing the generated caption.
    """
    
    result_caption = []
    with torch.no_grad():
        x = encoder(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states =  decoder.lstm(x, states)
            output = decoder.linear(hiddens.squeeze(0))
            predicted = output.argmax(dim=0)    
            result_caption.append(predicted.item())
            x = decoder.embedding(predicted).unsqueeze(0)
            if vocabulary[str(predicted.item())] == "<EOS>":
                break
    result = [vocabulary[str(idx)] for idx in result_caption]
    return result