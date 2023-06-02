import os
import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_img1 = transform(Image.open("test_examples/girl.jpeg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: A girl running in the field")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )

def save_checkpoint(state, step, checkpoint_dir="./checkpoint", filename="my_checkpoint.pth.tar"):
    os.mkdir("checkpoint/")
    print("=> Saving checkpoint")
    torch.save(state, f"{checkpoint_dir}/{step}_{filename}")

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step