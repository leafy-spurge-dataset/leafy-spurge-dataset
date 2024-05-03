from leafy_spurge_dataset.plot import (
    plot, add_plot_args
)
from leafy_spurge_dataset.train import (
    train, add_train_args
)

import argparse
import os


def quickstart(args: argparse.Namespace):

    input_file_name = os.path.join(
        os.path.dirname(__file__),
        "train.py",
    )

    with open(input_file_name, "r") as g:
        with open(args.output_train_file_name, "w") as f:
            f.write(g.read())

    input_file_name = os.path.join(
        os.path.dirname(__file__),
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
}


def entry_point():

    parser = argparse.ArgumentParser(
        "Leafy Spurge Dataset Command Line Interface"
    )
    
    subparsers = parser.add_subparsers()

    add_plot_args(subparsers.add_parser("plot"))
    add_train_args(subparsers.add_parser("train"))
    add_quickstart_args(subparsers.add_parser("quickstart"))

    args = parser.parse_args()
    FUNCTIONS[args.command_name](args)