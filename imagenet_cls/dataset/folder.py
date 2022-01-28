from copy import deepcopy
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

from PIL import Image, ImageCms
import numpy as np

import random
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from multiprocessing import Manager

from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    split: str = 'train',
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    if split == 'test':
        instances_path = f'PATH_TO/imgnet_instances_{split}.torch'       # for cache and speed up data loading
    else:
        instances_path = f'PATH_TO/imgnet_instances_train.torch'
    if os.path.isfile(instances_path):
        return torch.load(instances_path)

    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in tqdm(sorted(class_to_idx.keys())):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    torch.save(instances, instances_path)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_image_num: Optional[int] = None,
            split: Optional[str] = None,
            seed: Optional[int] = None,
            cache: bool = False,
            rgb2lab: Optional[bool] = False
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        print('Making dataset')
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, split)
        print('Made dataset')
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        # Modify to allow using max_image_num
        self.max_image_num = max_image_num
        self.split = split
        self.seed = seed
        self.rgb2lab = rgb2lab
        self.shared_cache = Manager().dict() if cache else None

        # samples contain all the image paths and labels [(path, label),(path, label), ...]
        # do stratified selection for training data
        if self.max_image_num is not None:
            # shuffle all the samples with a fixed seed
            random.seed(self.seed)
            
            # get all the labels
            y = []
            for samp in samples:
                y.append(samp[1])

            # shuffle the labels
            random.shuffle(y)

            # get the total max image nums
            if self.split == 'train':
                total_image_num = int(self.max_image_num / 0.8)
            elif self.split == 'val':
                total_image_num = int(self.max_image_num / 0.2)
            
            print("Total image num: ", total_image_num)
            ssplit = StratifiedShuffleSplit(n_splits=1, train_size=int(total_image_num*0.8), test_size=int(total_image_num*0.2), random_state=self.seed)
            train_indices, test_indices = next(ssplit.split(X=np.zeros(len(y)), y=y))

            # get samples by indices
            # np.array will change the label type from int to string
            samples = np.array(samples)
            if self.split == 'train':
                samples = samples[train_indices].tolist()
            elif self.split == 'val':
                samples = samples[test_indices].tolist()

            # convert the labels from string back to int
            samples = [(s[0], int(s[1])) for s in samples]

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.max_image_num:
            if self.split == 'train':
                print("Using {} training images.".format(len(self.samples)))
            elif self.split == 'val':
                print("Using {} validation images.".format(len(self.samples)))
        else:
            print("Using {} test images.".format(len(self.samples)))


    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        split: str = 'train',
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, split=split)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.shared_cache is not None and path in self.shared_cache:
            sample  = self.shared_cache[path]
        else:
            sample = self.loader(path)
            if self.shared_cache is not None:
                self.shared_cache[path] = deepcopy(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.rgb2lab:
            # convert rgb to lab
            res = convertrgb2lab(sample)
    
            # For this L channel image, do not use default transform
            # use the following transform to transform it into [-50, 50]
            res = np.array(res)
            res = res.astype(np.float32)
            # convert to 0-100 first, and then subtract 50.0
            # TODO: check if normalize to [0, 1] or [-50, 50]
            res = (res * (100.0 / 255.0)) - 50.0

            # from numpy array to tensor
            res = torch.from_numpy(res).float()

            # add one channel
            sample = res.unsqueeze(0)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def convertrgb2lab(img):
    # convert RGB PIL image to Lab, return the L channel, range: 0-255
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(img, rgb2lab)
    L, a, b = Lab.split()
    return L


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            max_image_num: Optional[int] = None,
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            seed: Optional[int] = None,
            cache: bool = False,
            rgb2lab: Optional[bool] = False,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          max_image_num=max_image_num,
                                          split=split,
                                          seed=seed,
                                          cache=cache,
                                          rgb2lab=rgb2lab,
                                          )
        self.imgs = self.samples