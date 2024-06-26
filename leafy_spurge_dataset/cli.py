from leafy_spurge_dataset.experiments.plot import (
    plot, add_plot_args
)
from leafy_spurge_dataset.experiments.train_classifier import (
    train, add_train_args
)
from leafy_spurge_dataset.experiments.evaluate_openai import (
    evaluate, add_evaluate_args
)

import argparse
import os


def quickstart(args: argparse.Namespace):

    input_file_name = os.path.join(
        os.path.dirname(__file__),
        "experiments",
        "train_classifier.py",
    )

    with open(input_file_name, "r") as g:
        with open(args.output_train_file_name, "w") as f:
            f.write(g.read())

    input_file_name = os.path.join(
        os.path.dirname(__file__),
        "experiments",
        "evaluate_openai.py",
    )

    with open(input_file_name, "r") as g:
        with open(args.output_evaluate_file_name, "w") as f:
            f.write(g.read())

    input_file_name = os.path.join(
        os.path.dirname(__file__),
        "experiments",
        "plot.py",
    )

    with open(input_file_name, "r") as g:
        with open(args.output_plot_file_name, "w") as f:
            f.write(g.read())


def add_quickstart_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--output_train_file_name",
        type=str,
        default="leafy_spurge_train_classifier.py",
        help="Name of the output file",
    )

    parser.add_argument(
        "--output_evaluate_file_name",
        type=str,
        default="leafy_spurge_evaluate_openai.py",
        help="Name of the output file",
    )

    parser.add_argument(
        "--output_plot_file_name",
        type=str,
        default="leafy_spurge_plot_results.py",
        help="Name of the output file",
    )

    parser.set_defaults(
        command_name="quickstart",
    )


FUNCTIONS = {
    "quickstart": quickstart,
    "plot": plot,
    "train": train,
    "evaluate": evaluate,
}


def entry_point():

    parser = argparse.ArgumentParser(
        "Leafy Spurge Dataset Command Line Interface"
    )
    
    subparsers = parser.add_subparsers()

    add_plot_args(subparsers.add_parser("plot"))
    add_train_args(subparsers.add_parser("train"))
    add_quickstart_args(subparsers.add_parser("quickstart"))
    add_evaluate_args(subparsers.add_parser("evaluate"))

    args = parser.parse_args()
    FUNCTIONS[args.command_name](args)
