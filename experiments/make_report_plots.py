from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("experiments/report_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_auroc_comparison():

    # Frame-level AUROC comparison between:
    # 1. Original hard Top-K MIL baseline
    # 2. AIS soft temporal weighting approach

    names = ["Top-K MIL\nBaseline", "AIS Soft\nSelection"]

    values = [0.7714, 0.7898]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(names, values)

    # Add AUROC values above bars.
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.4f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title(
        "AIS Improves Frame-Level Anomaly Localisation",
        fontsize=13,
        fontweight="bold",
    )

    plt.ylabel("Frame-level AUROC")

    plt.ylim(0.74, 0.82)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "figure_1_frame_auroc_comparison.png",
        dpi=220,
    )

    plt.close()


def plot_temperature_schedule():

    # AIS temperature annealing:
    # Early training -> broader segment contribution.
    # Later training -> stronger focus on confident segments.

    tau_initial = 1.0

    tau_final = 0.07

    tau_decay_epochs = 50

    total_epochs = 100

    epochs = np.arange(1, total_epochs + 1)

    taus = []

    for epoch in epochs:

        if epoch >= tau_decay_epochs:

            tau = tau_final

        else:

            ratio = epoch / tau_decay_epochs

            tau = tau_initial * (
                (tau_final / tau_initial) ** ratio
            )

        taus.append(tau)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        taus,
        linewidth=2,
    )

    plt.title(
        "AIS Gradually Focuses on High-Confidence Temporal Segments",
        fontsize=13,
        fontweight="bold",
    )

    plt.xlabel("Training epoch")

    plt.ylabel("Temperature τ")

    plt.grid(alpha=0.3)

    # Early training behaviour.
    plt.text(
        8,
        0.70,
        "High τ early:\nsegments contribute more broadly",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    # Later training behaviour.
    plt.text(
        58,
        0.15,
        "Low τ later:\nfocus shifts toward\nhigh-scoring segments",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "figure_2_ais_temperature_schedule.png",
        dpi=220,
    )

    plt.close()


def plot_selected_ais_score_curve():

    # Load saved AIS anomaly scores produced during evaluation.
    scores_path = Path(
        "experiments/ais/results/video_scores_ais.npy"
    )

    # Selected example video with meaningful variation
    # in temporal anomaly scores.
    video_name = "Arrest001_x264"

    # Display-friendly filename for the plot title.
    display_video_name = "Arrest001_x264.mp4"

    # Load saved dictionary:
    # {video_name -> segment anomaly scores}
    data = np.load(
        scores_path,
        allow_pickle=True,
    ).item()

    # Segment-level anomaly scores for this video.
    scores = np.array(
        data[video_name],
        dtype=float,
    )

    # Temporal segment indices:
    # 0, 1, 2, ..., 31
    x = np.arange(len(scores))

    # Original hard MIL baseline used Top-K = 8.
    top_k = 8

    # Select indices of highest-scoring segments.
    # These are the segments that the original
    # hard Top-K MIL would learn from.
    top_k_indices = np.argsort(scores)[-top_k:]

    top_k_scores = scores[top_k_indices]

    plt.figure(figsize=(9, 5))

    # AIS score curve:
    # AIS maintains anomaly scores across all segments.
    plt.plot(
        x,
        scores,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="AIS scores across all segments",
    )

    # Hard Top-8 baseline:
    # only selected segments contribute to learning.
    plt.scatter(
        top_k_indices,
        top_k_scores,
        s=120,
        marker="x",
        color="green",
        linewidths=3,
        label="Segments selected by hard Top-8 MIL",
        zorder=5,
    )

    plt.title(
        f"Example Video ({display_video_name})\n"
        "Baseline Hard Top-8 Selection vs AIS Soft Temporal Weighting",
        fontsize=13,
        fontweight="bold",
    )

    plt.xlabel("Temporal segment index")

    plt.ylabel("Predicted anomaly score")

    plt.ylim(0, 1.05)

    plt.grid(alpha=0.3)

    # Interpretation:
    # Hard Top-K MIL uses only a few selected segments,
    # whereas AIS retains anomaly scores across all segments.
    plt.text(
        0.02,
        0.68,
        "Hard Top-8 MIL uses only a few selected segments.\n"
        "AIS retains anomaly scores across all temporal segments,\n"
        "instead of ignoring the remaining video context.",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.legend(loc="lower right")

    plt.tight_layout()

    save_name = (
        f"figure_3_top8_vs_ais_score_curve_{video_name}.png"
    )

    plt.savefig(
        OUT_DIR / save_name,
        dpi=220,
    )

    plt.close()


def main():

    plot_auroc_comparison()

    plot_temperature_schedule()

    plot_selected_ais_score_curve()

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()