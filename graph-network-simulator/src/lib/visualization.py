import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

TYPE_TO_COLOR = {
    0: "green",
    1: "blue",
    2: "orange",
}


def visualize_prepare(ax, position, bounds=[[-1000, 1000], [-1000, 1000]]):
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def plot_animation(cell_type, position_gt, position_pred=None, bounds=[[-1000, 1000], [-1000, 1000]], interval=10, step=1):
    fig, axes = plt.subplots(1, 2 if position_pred else 1, figsize=(10, 5))
    if not isinstance(axes, list):
        axes = [axes]
    
    plot_info = []
    plot_info.append(visualize_prepare(axes[0], position_gt, bounds))
    axes[0].set_title("Ground truth")
    if position_pred:
        plot_info.append(visualize_prepare(axes[1], position_pred, bounds))
        axes[1].set_title("Prediction")

    plt.close()

    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for _type, line in points.items():
                mask = (cell_type[:, step_i] == _type)
                line.set_data(position[mask, step_i, 0], position[mask, step_i, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, position_gt.shape[1], step=step),
        interval=interval,
        blit=True,
    )