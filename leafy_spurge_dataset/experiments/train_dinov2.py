from torch.utils.data import (
    DataLoader,
    random_split,
)

import argparse
import os
import numpy as np
import random

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import v2
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from transformers import (
    Dinov2ForImageClassification,
)


def train(args: argparse.Namespace):

    from leafy_spurge_dataset import LeafySpurgeDataset

    os.makedirs(args.output_dir, exist_ok=True)

    # set the seed for reproducibility

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data augmentation and normalization

    train_transform = v2.Compose([
        v2.RandomApply(
            [v2.ColorJitter(
                brightness=0.8,
                contrast=0.7,
                saturation=0,
                hue=0,
            )],
            p=0.5,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=90)], p=0.5),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # load the Leafy Spurge dataset

    train_dataset = LeafySpurgeDataset(
        version=args.dataset_version,
        split='train',
        transform=train_transform,
        output_dict=False,
        examples_per_class=args.examples_per_class,
        seed_subset=args.seed,
        invert_subset=False,
    )
    
    if args.examples_per_class is not None:

        val_dataset = LeafySpurgeDataset(
            version=args.dataset_version,
            split='train',
            transform=test_transform,
            output_dict=False,
            examples_per_class=args.examples_per_class,
            seed_subset=args.seed,
            invert_subset=True,
        )

    else:  # use 20% of the training data for validation

        train_dataset_size = len(train_dataset)
        val_split_size = int(0.2 * train_dataset_size)
        train_split_size = train_dataset_size - val_split_size

        train_dataset, val_dataset = random_split(
            train_dataset, (train_split_size, val_split_size))

    test_dataset = LeafySpurgeDataset(
        version=args.dataset_version,
        split='test',
        transform=test_transform,
        output_dict=False,
    )

    # create dataloaders for training, validation, and testing

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # load the DINOv2 model

    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base",
    )

    # configure the PEFT model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "query",
            "key",
            "value",
            "dense",
        ],
    )

    model = get_peft_model(model, lora_config)

    # modify the classifier head

    model.num_labels = train_dataset.num_classes
    model.classifier = nn.Linear(
        model.config.hidden_size * 2,
        model.num_labels,
    )

    # set the learning rate and optimizer

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # use Huggingface's Accelerate library

    accelerator = Accelerator()

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # record the training progress

    dataframe_records = []

    for epoch in range(args.num_epochs):

        # train the model

        model.train()

        train_accuracy = 0.0

        for images, labels in train_dataloader:

            # accumulate gradients

            with accelerator.accumulate(model):

                images = images.to(accelerator.device)
                labels = labels.to(accelerator.device)

                # forward pass with automatic mixed precision

                with accelerator.autocast():

                    outputs = model(
                        pixel_values=images,
                        labels=labels,
                    )

                # backward pass

                accelerator.backward(outputs.loss)
                optimizer.step()
                optimizer.zero_grad()

            # calculate the accuracy

            predictions = outputs.logits.argmax(dim=-1)
            accuracy = predictions == labels
            accuracy = accuracy.float().sum().item()

            train_accuracy += accuracy

        train_accuracy = (
            train_accuracy / 
            len(train_dataset)
        )

        if accelerator.is_main_process:

            print("Epoch {:05d} Train Accuracy: {:0.5f}".format(
                epoch, train_accuracy
            ))

        # validate the model

        model.eval()

        val_accuracy = 0.0

        for images, labels in val_dataloader:

            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)

            # forward pass with no gradient computation

            with torch.no_grad():

                outputs = model(
                    pixel_values=images,
                    labels=labels,
                )

            # calculate the accuracy

            predictions = outputs.logits.argmax(dim=-1)
            accuracy = predictions == labels
            accuracy = accuracy.float().sum().item()

            val_accuracy += accuracy

        val_accuracy = (
            val_accuracy / 
            len(val_dataset)
        )

        if accelerator.is_main_process:

            print("Epoch {:05d} Val Accuracy: {:0.5f}".format(
                epoch, val_accuracy
            ))

        # record the training progress

        dataframe_records.append({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "method": "DINOv2", 
        })

    # test the model

    test_accuracy = 0.0

    for images, labels in test_dataloader:

        images = images.to(accelerator.device)
        labels = labels.to(accelerator.device)

        # forward pass with no gradient computation

        with torch.no_grad():

            outputs = model(
                pixel_values=images,
                labels=labels,
            )

        # calculate the accuracy

        predictions = outputs.logits.argmax(dim=-1)
        accuracy = predictions == labels
        accuracy = accuracy.float().sum().item()

        test_accuracy += accuracy

    test_accuracy = (
        test_accuracy / 
        len(test_dataset)
    )

    if accelerator.is_main_process:

        print("Final Test Accuracy: {:0.5f}".format(
            test_accuracy
        ))

    # record the training progress

    dataframe_records.append({
        "epoch": args.num_epochs,
        "test_accuracy": test_accuracy,
        "method": "DINOv2", 
    })

    # wait for all processes to finish

    accelerator.wait_for_everyone()

    # save the training progress and the model

    if accelerator.is_main_process:

        # save the training progress

        df = pd.DataFrame(
            dataframe_records
        )
        
        df.to_csv(
            os.path.join(
                args.output_dir,
                "data.csv",
            ),
            index=False,
        )

        # save the model

        model = accelerator.unwrap_model(
            model
        )

        model.save_pretrained(
            os.path.join(
                args.output_dir,
                "model",
            ),
        )


def add_train_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--dataset_version",
        type=str,
        default="crop",
        help="Version of the dataset to use",
        choices=["crop", "context"],
    )

    parser.add_argument(
        "--examples_per_class",
        type=int,
        default=None,
        help="Number of examples per class",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in a batch",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the model",
    )

    parser.set_defaults(
        command_name="train",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Starter code for training Leafy Spurge classifiers"
    )

    add_train_args(parser)

    train(parser.parse_args())