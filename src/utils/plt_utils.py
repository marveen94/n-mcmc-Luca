import typing


from typing import List, Tuple, Optional

import numpy as np
from matplotlib.axes import Axes


def plt_hist(
    ax: Axes,
    data: np.ndarray,
    labels: List[str],
    colors: List[str],
    bins: int,
    ground_state: float,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ticklables: Optional[List[float]] = None,
    text: Optional[Tuple[float, float, str]] = None,
):
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.set_xlabel(r"$H/N$")

    bins = np.linspace(data.min(), data.max(), num=bins).tolist()

    for i in range(data.shape[-1]):
        if i == data.shape[-1] - 1:
            _ = ax.hist(
                data[:, i],
                bins=bins,
                label=f"{labels[i]}",
                histtype="bar",
                edgecolor="k",
                color="lightgrey",
                linewidth=1,
                alpha=0.5,
            )
            continue

        _ = ax.hist(
            data[:, i],
            bins=bins,
            label=f"{labels[i]}",
            # histtype="bar",
            linewidth=0.1,
            edgecolor="k",
            alpha=0.9 - i * 0.1,
            color=colors[i],
        )

    ax.vlines(
        ground_state,
        0,
        data[:, i].shape[-1] * 0.8,
        linewidth=4.0,
        colors="red",
        linestyles="dashed",
        alpha=0.7,
        label="GS",
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if ticklables is not None:
        ax.tick_params(axis="x", which="minor", bottom=True)
        ax.locator_params(axis="x", nbins=len(ticklables))
        ax.set_xticklabels(ticklables)

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    ax.legend(framealpha=0.8)
    return


def plt_engbeta(
    ax: Axes,
    data: np.ndarray,
    labels: List[str],
    colors: List[str],
    markers: List[str],
    markersize: List[int],
    fillstyle: List[str],
    ground_state: float,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    text: Optional[Tuple[float, float, str]] = None,
):
    if isinstance(data, list):
        for i, (d, l, c, m, ms, fs) in enumerate(
            zip(data, labels, colors, markers, markersize, fillstyle)
        ):
            ax.errorbar(
                d[:, 0],
                d[:, 1],
                yerr=d[:, 2],
                linewidth=0.1,
                elinewidth=2,
                markeredgewidth=2,
                label=l,
                color=c,
                marker=m,
                mec=c,
                markersize=ms,
                fillstyle=fs,
            )
    else:
        for i, (l, c, m, ms, fs) in enumerate(
            zip(labels, colors, markers, markersize, fillstyle)
        ):
            ax.errorbar(
                data[:, 0],
                data[:, i * 2 + 1],
                yerr=data[:, i * 2 + 2],
                linewidth=0.1,
                elinewidth=2,
                markeredgewidth=2,
                label=l,
                color=c,
                marker=m,
                mec=c,
                markersize=ms,
                fillstyle=fs,
            )
    ax.hlines(
        ground_state,
        xmin=xlim[0] if xlim is not None else data[:, 0].min(),
        xmax=xlim[1] if xlim is not None else data[:, 0].max(),
        colors="red",
        linestyles="dashed",
        label="GS",
        linewidth=4,
    )

    ax.set_ylabel(r"$E/N$")
    ax.set_xlabel(r"$\mathrm{\beta}$")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    ax.legend(framealpha=0.8)
    return


def plt_ar(
    ax: Axes,
    data: np.ndarray,
    line_styles: List[str],
    labels: List[str],
    colors: List[str],
    markersize: int = 6,
    text: Optional[Tuple[float, float, str]] = None,
):
    for i, (ls, l, c) in enumerate(zip(line_styles, labels, colors)):
        ax.plot(
            data[:, 0],
            data[:, i + 1],
            ls,
            color=c,
            label=l,
            markersize=markersize,
        )

    ax.set_ylabel(r"$\mathrm{A_r}[\%]$")
    ax.set_xlabel(r"$\mathrm{\beta}$")

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    ax.legend(framealpha=0.8)
    return


def plt_acf(
    ax: Axes,
    data: np.ndarray,
    linestyles: List[str],
    labels: List[str],
    colors: List[str],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    text: Optional[Tuple[float, float, str]] = None,
):
    for i, (ls, l, c) in enumerate(zip(linestyles, labels, colors)):
        ax.plot(np.arange(data.shape[0]) + 1, data[:, i], ls, label=l, color=c)

    ax.set_ylabel(r"$\mathrm{c(\tau)}$")
    ax.set_xlabel(r"$\mathrm{\tau}$")

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax.set_xscale("log")
    ax.set_yscale("log")

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    ax.legend(framealpha=0.8, fontsize=19)
    return


def plt_engstep(
    ax: Axes,
    data: np.ndarray,
    labels: List[str],
    colors: List[str],
    xlim: Tuple[float, float],
    ground_state: float,
    text: Optional[Tuple[float, float, str]] = None,
):
    SKIP = 8000
    STEP = 30

    for file, label, color in zip(data, labels, colors):
        if file == "ssf" or file == "pt":
            ax.plot(
                np.concatenate(
                    (np.arange(SKIP), np.arange(data[file].shape[0])[SKIP::STEP])
                )
                + 1,
                np.concatenate((data[file][:, 0][:SKIP], data[file][:, 0][SKIP::STEP]))
                if file == "ssf"
                else np.concatenate((data[file][:SKIP], data[file][SKIP::STEP])),
                label=label,
                color=color,
                linewidth=4.0 if file == "hmc" else 1.5,
                alpha=0.7 if file == "hmc" else 1,
            )
        elif file == "hmc":
            ax.plot(
                np.arange(data[file].shape[0]) + 1,
                data[file],
                label=label,
                color=color,
                linewidth=4.0 if file == "hmc" else 1.5,
                alpha=0.7 if file == "hmc" else 1,
            )
        if file == "ssf":
            ax.fill_between(
                np.concatenate(
                    (np.arange(SKIP), np.arange(data[file].shape[0])[SKIP::STEP])
                )
                + 1,
                np.concatenate((data[file][:, 0][:SKIP], data[file][:, 0][SKIP::STEP]))
                + np.concatenate(
                    (data[file][:, 1][:SKIP], data[file][:, 1][SKIP::STEP])
                ),
                np.concatenate((data[file][:, 0][:SKIP], data[file][:, 0][SKIP::STEP]))
                - np.concatenate(
                    (data[file][:, 1][:SKIP], data[file][:, 1][SKIP::STEP])
                ),
                alpha=0.1,
                color=color,
            )
    ax.hlines(
        ground_state,
        xmin=0,
        xmax=xlim[1],
        colors="red",
        linestyles="dashed",
        label="GS",
        linewidth=3.0,
    )

    ax.set_xscale("log")
    ax.set_ylabel(r"$H/N$")
    ax.set_xlabel(r"$\mathrm{\tau}$")

    ax.set_xlim(xlim)
    ax.set_ylim((ground_state - 0.01, data["ssf"][..., 0].max()))

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    ax.legend(framealpha=0.8)
    return


def plt_chains(
    ax: Axes,
    data: np.ndarray,
    strengths: np.ndarray,
    dwave_default: List[float],
    ground_state: float,
    xlim: Tuple[float, float],
    legend: bool = True,
    text: Optional[Tuple[float, float, str]] = None,
):
    ax.errorbar(
        strengths,
        data[:, 0],
        data[:, 1],
        capsize=5.0,
        elinewidth=2.5,
        linewidth=0.5,
        marker="s",
        color="tab:orange",
        fillstyle="none",
        markersize=8,
        markeredgewidth=2,
        label=r"$E_{\mathrm{avg}}/N$",
    )

    ax.plot(
        dwave_default[0],
        dwave_default[3],
        "d",
        markersize=10,
        color="tab:green",
        mec="tab:green",
        label=r"Default $E_{\mathrm{min}}/N$",
    )
    ax.errorbar(
        dwave_default[0],
        dwave_default[1],
        dwave_default[2],
        capsize=5.0,
        elinewidth=1.5,
        linewidth=0.1,
        marker="d",
        color="tab:green",
        mec="tab:green",
        fillstyle="none",
        markersize=10,
        markeredgewidth=2,
        label=r"Default $E_{\mathrm{avg}}/N$",
    )

    ax.plot(strengths, data[:, 2], "-s", label=r"$E_{\mathrm{min}}/N$", linewidth=1.0)

    ax.hlines(
        ground_state,
        xmin=xlim[0] - 0.4,
        xmax=xlim[1] + 0.4,
        colors="red",
        linestyles="dashed",
        label="GS",
        linewidth=3,
    )

    ax.minorticks_off()
    ax.set_xlim(xlim)

    ax.set_xlabel(r"$J_c$")
    ax.set_ylabel(r"$E/N$")

    if text is not None:
        ax.text(text[0], text[1], text[2], fontsize="x-large")

    if legend:
        ax.legend(loc="upper left", framealpha=0.8)
    return
