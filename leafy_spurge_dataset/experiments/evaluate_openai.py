from openai import OpenAI
from PIL import Image

from collections import defaultdict
from itertools import product

from io import BytesIO

import pandas as pd

import base64
import os
import glob

import argparse
import tqdm


client = OpenAI()


def evaluate(args: argparse.Namespace):

    from leafy_spurge_dataset import LeafySpurgeDataset

    # load the Leafy Spurge dataset

    train_dataset = LeafySpurgeDataset(
        version = args.dataset_version,
        split = 'train',
        output_dict = True,
        examples_per_class = args.examples_per_class,
        seed_subset = args.seed,
    )

    # create few-shot examples for the openai chat model

    few_shot_examples = []

    for train_idx in range(len(train_dataset)):

        train_dict = train_dataset[train_idx]

        # load and resize the image to the desired resolution

        pil_image = train_dict["image"]
        pil_image = pil_image.resize(
            (args.image_resolution, args.image_resolution))
        pil_image = pil_image.convert("RGB")

        # convert the image to base64 encoding

        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")

        image_data = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # get the label of the image in text

        label = train_dict["label"]
        label_name = train_dataset.class_id_to_name[label]

        # show an image from the training dataset

        few_shot_examples.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Does this image contain Leafy Spurge? "
                    "Answer either yes or no, with no punctuation."
                )
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }]
        })

        # show the label of the image

        few_shot_examples.append({
            "role": "assistant",
            "content": "yes" if label_name == "present" else "no",
        })

    # load the test dataset

    test_dataset = LeafySpurgeDataset(
        version = args.dataset_version,
        split = 'test',
        output_dict = True,
        examples_per_class = args.evaluations_per_class,
        seed_subset = args.seed,
    )

    os.makedirs(args.output_dir, exist_ok = True)

    # evaluate the model on the test dataset

    dataframe_records = []

    for test_idx in (pbar := tqdm.trange(len(test_dataset))):

        test_dict = test_dataset[test_idx]

        # load and resize the image to the desired resolution

        pil_image = test_dict["image"]
        pil_image = pil_image.resize(
            (args.image_resolution, args.image_resolution))
        pil_image = pil_image.convert("RGB")

        # convert the image to base64 encoding

        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")

        image_data = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # get the label of the image in text

        label = test_dict["label"]
        label_name = test_dataset.class_id_to_name[label]

        # ask the model to predict the label of the image

        completion = client.chat.completions.create(
            model=args.openai_model_name,
            messages=[{
                "role": "system",
                "content": (
                    "Thanks for helping us with our weed annotation project! "
                    "You are an expert in ecology and plant species. "
                    "We will show you aerial drone images and then ask whether "
                    "each image contains the invasive plant Leafy Spurge, also called Euphorbia esula. "
                    "Your answers will help us remove this invasive plant, so do your best!"
                )
            }, *few_shot_examples, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "Does this image contain Leafy Spurge? "
                        "Answer either yes or no, with no punctuation."
                    )
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }]
            }]
        )

        response = completion.choices[0].message.content
        response = response.strip(" .,!?").lower()

        # calculate the accuracy of the model

        test_accuracy = (1 if (response == "yes" and label_name == "present") 
            or (response == "no" and label_name == "absent") else 0)

        # log model predictions on this test datapoint

        dataframe_records.append({
            "version": args.dataset_version,
            "model": args.openai_model_name,
            "examples_per_class": args.examples_per_class,
            "image_resolution": args.image_resolution,
            "test_idx": test_idx,
            "test_accuracy": test_accuracy,
            "label": label_name})

        pbar.set_description(
            "Concept: {} | Label: {} | Accuracy: {}".format(
                label_name, response, test_accuracy))

    # save the evaluation results

    dataframe = pd.DataFrame(
        dataframe_records)
    
    dataframe.to_csv(
        os.path.join(
            args.output_dir,
            "data.csv",
        ),
        index = False,
    )


def add_evaluate_args(parser: argparse.ArgumentParser):

    # Add arguments to the parser to evaluate the model

    parser.add_argument(
        "--openai_model_name",
        type = str,
        default = "gpt-4o",
        help = "Model name from OpenAI to evaluate",
        choices = ["gpt-4o", "gpt-4-turbo"],
    )

    parser.add_argument(
        "--dataset_version",
        type = str,
        default = "crop",
        help = "Version of the dataset to use",
        choices = ["crop", "context"],
    )

    parser.add_argument(
        "--examples_per_class",
        type = int,
        default = 1,
        help = "Number of examples per class",
    )

    parser.add_argument(
        "--evaluations_per_class",
        type = int,
        default = 32,
        help = "Number of test examples per class",
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducibility",
    )
    
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./openai_evaluation",
        help = "Directory to save the model",
    )
    
    parser.add_argument(
        "--image_resolution",
        type = int,
        default = 256,
        help = "Size of image to pass to openai model",
    )

    parser.set_defaults(
        command_name = "evaluate",
    )


if __name__ ==  "__main__":

    parser = argparse.ArgumentParser(
        "Starter code for evaluating openai models on Leafy Spurge Dataset"
    )

    add_evaluate_args(parser)

    evaluate(parser.parse_args())
