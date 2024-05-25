import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import argparse
import os
import glob


def plot(args: argparse.Namespace):

    os.makedirs(
        os.path.dirname(args.plot_file_name),
        exist_ok=True,
    )

    glob.glob(args.csv_file_name)

    data = pd.concat([
        pd.read_csv(file_name)
        for file_name in glob.glob(args.csv_file_name)
    ], ignore_index=True)

    color_palette = sns.color_palette("colorblind")

    matplotlib.rc('font', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

    plt.rcParams['text.usetex'] = False
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axis = plt.subplots(
        1, # nrows 
        1, # ncols
        figsize=(
            8.0, # width
            6.0, # height
        ),
    )

    g = sns.lineplot(
        x="epoch",
        y="val_accuracy",
        hue="method",
        data=data,
        errorbar=('ci', 95),
        linewidth=4,
        palette=color_palette,
        ax=axis,
    )

    handles, labels = axis.get_legend_handles_labels()
    axis.legend([],[], frameon=False)

    axis.set(xlabel=None)
    axis.set(ylabel=None)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.grid(
        color='grey',
        linestyle='dotted',
        linewidth=2
    )

    axis.set_title(
        "Leafy Spurge Classification",
        fontsize=24,
        fontweight='bold',
        pad=12,
    )

    axis.set_xlabel(
        "Epoch",
        fontsize=20,
        labelpad=12,
        fontdict=dict(weight='bold'),
    )

    axis.set_ylabel(
        "Accuracy (Val)",
        fontsize=20,
        labelpad=12,
        fontdict=dict(weight='bold')
    )
        
    axis.set_ylim(-0.1, 1.1)

    axis.set_xticks(
        range(
            0,
            data["epoch"].max() + 1,
            5,
        )
    )

    if len(labels) > 1:

        legend = fig.legend(
            handles, labels,
            loc="lower center",
            prop={'size': 24, 'weight': 'bold'}, 
            ncol=len(labels),
        )

        for i, x in enumerate(legend.legend_handles):
            x.set_linewidth(4.0)
            x.set_color(color_palette[i])
    
    plt.tight_layout()

    if len(labels) > 1:

        fig.subplots_adjust(
            bottom=0.3,
        )

    plt.savefig(
        args.plot_file_name,
        bbox_inches='tight',
    )


def add_plot_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="output/*.csv",
        help="The name of the csv file to plot",
    )

    parser.add_argument(
        "--plot_file_name",
        type=str,
        default="output/plot.png",
        help="The name of the plot file to create",
    )

    parser.set_defaults(
        command_name="plot",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Starter code for plotting Leafy Spurge classifier results"
    )

    add_plot_args(parser)

    plot(parser.parse_args())