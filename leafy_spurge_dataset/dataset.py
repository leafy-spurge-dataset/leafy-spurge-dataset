from typing import Optional, Callable, Tuple, Union, Dict, Any
from torch.utils.data import Dataset
from torchvision.datasets.vision import StandardTransform
from datasets import load_dataset


DATASET_PATH = 'mpg-ranch/leafy_spurge'

DEFAULT_VERSION = 'crop'
DEFAULT_SPLIT = 'train'


class LeafySpurgeDataset(Dataset):

    def __init__(self, version: str = DEFAULT_VERSION,
                 split: str = DEFAULT_SPLIT,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 output_dict: bool = False):
        
        super(LeafySpurgeDataset, self).__init__()

        self.output_dict = output_dict

        has_transforms = transforms is not None
        has_separate_transform = (
            transform is not None or 
            target_transform is not None)

        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed as argument"
            )

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        self.huggingface_dataset = load_dataset(
            DATASET_PATH,
            name=version,
            split=split,
        )

        self.class_names = self.huggingface_dataset.features['label'].names
        self.num_classes = len(self.class_names)

        self.class_name_to_id = {
            class_name: class_id
            for class_id, class_name in enumerate(self.class_names)
        }
        self.class_id_to_name = {
            class_id: class_name
            for class_id, class_name in enumerate(self.class_names)
        }

    def __len__(self) -> int:

        return len(self.huggingface_dataset)

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], Tuple[Any, Any]]:

        sample = self.huggingface_dataset[idx]

        image = sample.pop('image').convert('RGB')
        label = sample.pop('label')

        if self.transforms is not None:

            image, label = self.transforms(
                image, label,
            )

        if self.output_dict:

            return {
                'image': image,
                'label': label,
                **sample,
            }

        return image, label
