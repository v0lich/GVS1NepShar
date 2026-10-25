#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import plotly.graph_objects as go


def read_results(path: Path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    factors = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
    }

    cuda = {}
    eigen = {}

    for item in data["benchmarks"]:
        if item.get("run_type") != "iteration":
            continue

        parts = item["name"].split("/")
        if len(parts) < 2:
            continue

        name = parts[0]
        n = int(parts[1])
        unit = item["time_unit"]
        seconds = float(item["real_time"]) * factors[unit]

        if name == "BM_CudaVectorAdd":
            cuda[n] = seconds
        elif name == "BM_EigenVectorAdd":
            eigen[n] = seconds

    return cuda, eigen


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: plot_benchmarks.py BENCHMARKS_JSON OUTPUT_DIR"
        )

    cuda, eigen = read_results(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)

    sizes = sorted(set(cuda) & set(eigen))
    if not sizes:
        raise SystemExit("No matching CUDA/Eigen benchmark rows found")

    complexity = go.Figure()
    complexity.add_trace(
        go.Scatter(
            x=sizes,
            y=[cuda[n] for n in sizes],
            mode="lines+markers",
            name="CUDA Vector",
        )
    )
    complexity.add_trace(
        go.Scatter(
            x=sizes,
            y=[eigen[n] for n in sizes],
            mode="lines+markers",
            name="Eigen::VectorXf",
        )
    )

    complexity.update_layout(
        title="Реальная вычислительная сложность сложения",
        xaxis_title="Размер вектора n",
        yaxis_title="Время, с",
        xaxis_type="log",
        yaxis_type="log",
    )

    complexity.write_html(output / "real_complexity.html")

    speedup = [eigen[n] / cuda[n] for n in sizes]

    speedup_figure = go.Figure(
        go.Scatter(
            x=sizes,
            y=speedup,
            mode="lines+markers",
            name="Eigen / CUDA",
        )
    )

    speedup_figure.update_layout(
        title="Ускорение CUDA относительно Eigen::VectorXf",
        xaxis_title="Размер вектора n",
        yaxis_title="Ускорение, раз",
        xaxis_type="log",
    )

    speedup_figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="CUDA и Eigen равны",
    )

    speedup_figure.write_html(output / "speedup.html")


if __name__ == "__main__":
    main()
