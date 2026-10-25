#!/usr/bin/env python3
"""Графики реальной сложности и ускорения по результатам Google Benchmark.

usage: plot_benchmarks.py BENCHMARKS_JSON OUTPUT_DIR
"""

import json
import sys
from pathlib import Path

CUDA_NAME = "BM_CudaMatmul"
EIGEN_NAME = "BM_EigenMatmul"

UNIT_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def read_results(path: Path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    results = {CUDA_NAME: {}, EIGEN_NAME: {}}
    for item in data["benchmarks"]:
        if item.get("run_type") != "iteration":
            continue
        name, size = item["name"].split("/")[:2]
        if name in results:
            seconds = float(item["real_time"]) * UNIT_TO_SECONDS[item["time_unit"]]
            results[name][int(size)] = seconds

    return results[CUDA_NAME], results[EIGEN_NAME]


def print_table(sizes, cuda, eigen):
    # GFLOPS: умножение n x n требует 2 * n^3 операций (n^3 умножений + n^3 сложений).
    # Рост: во сколько раз выросло время при удвоении n (для O(n^3) ожидаем 8).
    print(f"{'n':>6} {'CUDA, мкс':>12} {'Eigen, мкс':>12} {'ускорение':>10} "
          f"{'CUDA GFLOPS':>12} {'Eigen GFLOPS':>13} {'рост CUDA':>10} {'рост Eigen':>11}")
    previous = None
    for n in sizes:
        flops = 2.0 * n**3
        growth_cuda = growth_eigen = ""
        if previous is not None:
            growth_cuda = f"{cuda[n] / cuda[previous]:.2f}"
            growth_eigen = f"{eigen[n] / eigen[previous]:.2f}"
        print(f"{n:>6} {cuda[n] * 1e6:>12.1f} {eigen[n] * 1e6:>12.1f} "
              f"{eigen[n] / cuda[n]:>10.2f} {flops / cuda[n] / 1e9:>12.1f} "
              f"{flops / eigen[n] / 1e9:>13.1f} {growth_cuda:>10} {growth_eigen:>11}")
        previous = n


def theoretical(sizes, measured):
    # Кривая c * n^3, проходящая через точку с наибольшим n.
    n_max = sizes[-1]
    c = measured[n_max] / n_max**3
    return [c * n**3 for n in sizes]


def save(figure, output: Path, stem: str):
    figure.write_html(output / f"{stem}.html")
    try:
        figure.write_image(output / f"{stem}.png", scale=2)
    except Exception as error:  # kaleido не установлен — оставляем только HTML
        print(f"PNG для {stem} не сохранён: {error}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_benchmarks.py BENCHMARKS_JSON OUTPUT_DIR")

    cuda, eigen = read_results(Path(sys.argv[1]))
    sizes = sorted(set(cuda) & set(eigen))
    if not sizes:
        raise SystemExit("No matching CUDA/Eigen benchmark rows found")

    print_table(sizes, cuda, eigen)

    import plotly.graph_objects as go

    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)

    complexity = go.Figure()
    complexity.add_trace(go.Scatter(
        x=sizes, y=[cuda[n] for n in sizes], mode="lines+markers",
        name="Matrix (CUDA, kernel_matmul_naive)"))
    complexity.add_trace(go.Scatter(
        x=sizes, y=[eigen[n] for n in sizes], mode="lines+markers",
        name="Eigen::MatrixXf"))
    complexity.add_trace(go.Scatter(
        x=sizes, y=theoretical(sizes, cuda), mode="lines",
        line=dict(dash="dot"), name="c·n³ (теория для CUDA)"))
    complexity.add_trace(go.Scatter(
        x=sizes, y=theoretical(sizes, eigen), mode="lines",
        line=dict(dash="dot"), name="c·n³ (теория для Eigen)"))
    complexity.update_layout(
        title="Реальная вычислительная сложность умножения матриц n×n",
        xaxis_title="Размер матрицы n", yaxis_title="Время, с",
        xaxis_type="log", yaxis_type="log", template="plotly_white",
        xaxis=dict(tickvals=sizes),
        yaxis=dict(dtick=1, exponentformat="power"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    save(complexity, output, "real_complexity")

    speedup = go.Figure(go.Scatter(
        x=sizes, y=[eigen[n] / cuda[n] for n in sizes], mode="lines+markers",
        name="Ускорение = t(Eigen) / t(CUDA)"))
    speedup.add_trace(go.Scatter(
        x=[sizes[0], sizes[-1]], y=[1.0, 1.0], mode="lines",
        line=dict(dash="dash", color="red"), name="CUDA и Eigen равны (1×)"))
    speedup.update_layout(
        title="Ускорение Matrix (CUDA) относительно Eigen::MatrixXf",
        xaxis_title="Размер матрицы n", yaxis_title="Ускорение, раз",
        xaxis_type="log", yaxis_type="log", template="plotly_white",
        xaxis=dict(tickvals=sizes),
        yaxis=dict(tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    save(speedup, output, "speedup")

    print(f"Графики сохранены в {output}")


if __name__ == "__main__":
    main()
