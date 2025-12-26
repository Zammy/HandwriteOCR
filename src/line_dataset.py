from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


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
                image_path = image_folder / f"line_{line_idx}.png"

                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    continue

                self.data_pairs.append((str(image_path), text))

        print(f"Loaded {len(self.data_pairs)} image-text pairs")

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path, text = self.data_pairs[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        return image, text


def test_dataset(num_samples: int = 5):
    dataset = LineImageTextDataset()

    print(f"\nTesting dataset with {num_samples} samples:\n")

    rng = np.random.default_rng()
    for _ in range(min(num_samples, len(dataset))):
        idx = int(rng.integers(0, len(dataset)))
        image, text = dataset[idx]

        img_display = image.numpy().squeeze()
        img_display = (img_display * 255).astype(np.uint8)

        cv2.imshow(f"Sample {idx}: {text}", img_display)
        print("  Press any key to continue...\n")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_dataset()
