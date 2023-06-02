import torch
import torchvision.transforms as transforms
from data.get_loader import get_loader
from models.model import EncoderDecoder
from PIL import Image

def inference(example_path, checkpoint_path, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    _, dataset = get_loader(
        image_folder="data/flickr8k/images",
        captions_file="data/flickr8k/captions.txt",
        transform=transform,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1

    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    test_img1 = transform(Image.open(example_path).convert("RGB")).unsqueeze(0).to(device)

    print("Example 1 CORRECT: A girl running in the field")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1, dataset.vocab))
    )

inference(example_path="test_examples/girl.jpeg", checkpoint_path="checkpoint/200_my_checkpoint.pth.tar")