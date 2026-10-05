#!/usr/bin/env python3
"""Build real-complexity and speedup plots from Google Benchmark JSON."""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go


def read_results(path: Path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    cuda = {}
    eigen = {}
    for item in data["benchmarks"]:
        if item.get("run_type") != "iteration":
            continue
        name = item["name"].split("/")[0]
        n = int(item["name"].split("/")[-1])
        seconds = float(item["real_time"])
        if name == "BM_CudaVectorAdd":
            cuda[n] = seconds
        elif name == "BM_EigenVectorAdd":
            eigen[n] = seconds
    return cuda, eigen


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_benchmarks.py BENCHMARKS_JSON OUTPUT_DIR")

    cuda, eigen = read_results(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(cuda) & set(eigen))
    if not sizes:
        raise SystemExit("No matching CUDA/Eigen benchmark rows found")

    complexity = go.Figure()
    complexity.add_trace(go.Scatter(x=sizes, y=[cuda[n] for n in sizes],
                                    mode="lines+markers", name="CUDA Vector"))
    complexity.add_trace(go.Scatter(x=sizes, y=[eigen[n] for n in sizes],
                                    mode="lines+markers", name="Eigen::VectorXf"))
    complexity.update_layout(title="Реальная вычислительная сложность сложения",
                             xaxis_title="Размер вектора n",
                             yaxis_title="Время, с", xaxis_type="log",
                             yaxis_type="log")
    complexity.write_html(output / "real_complexity.html")
    complexity.write_image(output / "real_complexity.png")

    speedup = [eigen[n] / cuda[n] for n in sizes]
    speedup_figure = go.Figure(go.Scatter(x=sizes, y=speedup,
                                          mode="lines+markers", name="Ускорение"))
    speedup_figure.update_layout(title="Ускорение CUDA относительно Eigen::VectorXf",
                                 xaxis_title="Размер вектора n",
                                 yaxis_title="Ускорение, раз", xaxis_type="log")
    speedup_figure.write_html(output / "speedup.html")
    speedup_figure.write_image(output / "speedup.png")


if __name__ == "__main__":
    main()
