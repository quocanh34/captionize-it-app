import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from data.get_loader import get_loader
from models.model import EncoderDecoder

def save_model(model, optimizer, step):

    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        step (int): Training step at which the model is saved.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    save_checkpoint(checkpoint, step)

def train():
    """
    Train the image captioning model.
    """
    # Initialize the transform
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Get dataloader and dataset
    train_loader, dataset = get_loader(
        image_folder="data/flickr8k/images",
        captions_file="data/flickr8k/captions.txt",
        transform=transform,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = False
    train_encoder = False
    print_every = 100

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 150


    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize model, loss etc
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_encoder

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    
    model.train()

    for epoch in range(num_epochs):
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs, captions = imgs.to(device), captions.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Feed forward
            outputs = model.forward(imgs, captions[:-1])

            # Calculate batch loss
            target = captions.reshape(-1)
            predict = outputs.reshape(-1, outputs.shape[2])
            loss = criterion(predict, target)

            # Write to tensorboard
            writer.add_scalar("Training loss", loss.item(), global_step=step)

            # Update step
            step += 1

            #Eval loss
            if step % print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))

                # Generate the caption
                model.eval()
                print_examples(model, device, dataset)

                # Back to train mode
                model.train()

            loss.backward(loss)
            optimizer.step()

            if step % 5000 == 0:
                save_model(model, optimizer, step)

if __name__ == "__main__":
    train()
