from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from src.data_augmentation import AugmentationConfig, get_augmentor


class LineImageTextDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "./data",
        images_dir: str = "./_inter_data",
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.data_pairs = []

        self._discover_and_load_pairs()

    def _discover_and_load_pairs(self):
        text_files = sorted(self.data_dir.glob("page*.txt"))
        pages = [f.stem for f in text_files]

        if not pages:
            print(f"Warning: No text files found in {self.data_dir}")
            return

        print(f"Discovered pages: {pages}")

        for page in pages:
            text_file = self.data_dir / f"{page}.txt"
            image_folder = self.images_dir / page

            if not text_file.exists():
                print(f"Warning: Text file not found: {text_file}")
                continue

            if not image_folder.exists():
                print(f"Warning: Image folder not found: {image_folder}")
                continue

            with open(text_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            for line_idx, text in enumerate(lines):
                text = LineImageTextDataset.clean_text(text)
                image_path = image_folder / f"line_{line_idx+1}.png"

                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    continue

                self.data_pairs.append((str(image_path), text))

        print(f"Loaded {len(self.data_pairs)} image-text pairs")
    
    @staticmethod
    def clean_text(text):
        # Replace non-breaking hyphen with standard hyphen
        text = text.replace('\u2011', '-') 
        # Optional: Replace other common "fancy" dashes
        text = text.replace('\u2013', '-').replace('\u2014', '-') 
        return text

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path, text = self.data_pairs[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if self.transform:
            image = self.transform(image)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        return image, text


def debug_augmentation_run(num_samples: int = 50):
    import time

    output_dir = Path("debug_outputs")
    output_dir.mkdir(exist_ok=True)

    aug_cfg = AugmentationConfig()
    dataset = LineImageTextDataset(
        transform=get_augmentor(aug_cfg),
    )

    print(f"Starting debug run. Saving {num_samples} images to {output_dir}/")

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img, text = dataset[idx]

        timestamp = str(int(time.time() * 1000))[-4:]
        filename = f"idx{idx}_{np.random.randint(1024)}_{timestamp}.png"

        img_display = img.numpy().squeeze()
        img_display = (img_display * 255).astype(np.uint8)

        cv2.imwrite(str(output_dir / filename), img_display)

    print(f"Done! Inspect files in {output_dir}")


if __name__ == "__main__":
    debug_augmentation_run()
