import argparse
import os
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing expanded EuroSAT.zip archive",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where the classification dataset will be written",
    )
    return parser


class _EuroSAT:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    IMAGE_FOLDER = "2750"
    TRAIN_SAMPLES = 1000
    VALID_SAMPLES = 500

    def __init__(self, input_path: str, output_path: str, split: str):
        self.input_path = input_path
        self.output_path = output_path
        self.split = split
        self.image_folder = os.path.join(self.input_path, self.IMAGE_FOLDER)
        self.images = []
        self.targets = []
        self.labels = sorted(os.listdir(self.image_folder))

        # There is no train/val split in the EUROSAT dataset, so we have to create it
        for i, label in enumerate(self.labels):
            label_path = os.path.join(self.image_folder, label)
            files = sorted(os.listdir(label_path))
            num_train = int(len(files) * 0.8)
            num_val = int(len(files) * 0.1)
            num_test = len(files) - num_train - num_val
            if split == 'train':
                self.images.extend(files[: num_train])
                self.targets.extend([i] * num_train)
            elif split == 'val':
                self.images.extend(
                    files[num_train : num_train + num_val]
                )
                self.targets.extend([i] * num_val)
            elif split == 'test':
                self.images.extend(
                    files[num_train + num_val:]
                )
                self.targets.extend([i] * num_test)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> bool:
        image_name = self.images[idx]
        target = self.labels[self.targets[idx]]
        image_path = os.path.join(self.image_folder, target, image_name)
        split_name = self.split
        shutil.copy(
            image_path, os.path.join(self.output_path, split_name, target, image_name)
        )
        return True


def create_disk_folder_split(dataset: _EuroSAT, split_path: str):
    """
    Create one split (example: "train" or "val") of the disk_folder hierarchy
    """
    for label in dataset.labels:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)
    loader = DataLoader(dataset, num_workers=36, batch_size=1, collate_fn=lambda x: x[0])
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


def create_euro_sat_disk_folder(input_path: str, output_path: str):
    """
    Read the EUROSAT dataset at 'input_path' and transform it to a disk folder at 'output_path'
    """
    print("Creating the training split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, output_path=output_path, split='train'),
        split_path=os.path.join(output_path, "train"),
    )
    print("Creating the validation split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, output_path=output_path, split='val'),
        split_path=os.path.join(output_path, "val"),
    )
    print("Creating the test split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, output_path=output_path, split='test'),
        split_path=os.path.join(output_path, "test"),
    )


if __name__ == "__main__":
    """
    Example usage:
    ```
    python create_euro_sat_data_files.py -i /path/to/euro_sat -o /output_path/to/euro_sat -d
    ```
    """
    args = get_argument_parser().parse_args()
    create_euro_sat_disk_folder(args.input, args.output)