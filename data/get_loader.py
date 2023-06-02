import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from data.vocab import Vocabulary

class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image and caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize the vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.to_list())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")
        
        # Transform the image
        if self.transform is not None:
            img = self.transform(img)

        # Generate numericalized caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption), caption
    
class MyCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
def get_loader(
        image_folder,
        captions_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    ):
    dataset = FlickrDataset(image_folder, captions_file, transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset

# if __name__ == "__main__":

#     transform = transforms.Compose(
#         [transforms.Resize((224, 224)), transforms.ToTensor(),]
#     )

#     loader, dataset = get_loader(image_folder="flickr8k/images", 
#                                  captions_file="flickr8k/captions.txt",
#                                  transform=transform,
#                                  batch_size=32,
#                                  num_workers=8,
#                                  shuffle=True,
#                                  pin_memory=True,
#                                 )
    
#     # for index, (img, caption) in enumerate(loader):
#     #     print(index)
#     #     print(img.shape)
#     #     print(caption)
#     #     print(caption.shape)
#     #     # for idx in caption[1]:
#     #     #     print(dataset.vocab.itos[idx.item()])
#     #     break 
#     print(dataset.__getitem__(150)[2])
#     for index in dataset.__getitem__(150)[1]:
#         print(dataset.vocab.itos[index.item()])