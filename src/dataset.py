from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NormalImageDataset(Dataset):
    """loading the normal images for autoencoder training"""
    def __init__(self,image_dir:str|Path,image_size:int=256):
        self.image_dir =Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f'{self.image_dir} does not exist')
        extensions={'png','jpg','jpeg','bnp','webp'}
        self.image_paths=sorted(path for path in self.image_dir.rglob("*")
                                if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        self.transform=transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index:int):
        image_path=self.image_path[index]
        with Image.open(image_path) as img:
            image=img.convert("RGB")
            image=self.transfrom(img)
        return image