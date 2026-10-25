# Практическая работа №2: умножение матриц на CUDA

Класс `Matrix` (фасад над `Data` + `MatrixView`), наивный кёрнел
`kernel_matmul_naive` на двумерной сетке нитей, `operator*`, тесты против
`Eigen::MatrixXf` и бенчмарки на CUDA Events.

## Структура

```
work2/
  core/        библиотека: Data, MatrixView, Matrix, кёрнел и operator*
  tests/       Google Test: 343 параметризованных теста (m, k, n) + граничные случаи
  benchmarks/  Google Benchmark: CUDA vs Eigen, n = 16 … 1024
  scripts/     построение графиков (plotly)
```

## Сборка

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Тесты

```bash
ctest --test-dir build --output-on-failure
# или подробно:
./build/tests/work2_tests
```

## Бенчмарки и графики

```bash
./build/benchmarks/work2_benchmarks \
  --benchmark_out=build/benchmarks.json \
  --benchmark_out_format=json
python3 -m pip install plotly kaleido
python3 scripts/plot_benchmarks.py build/benchmarks.json build/plots
```

Скрипт печатает таблицу (время, ускорение, GFLOPS, рост времени при удвоении n)
и сохраняет `real_complexity.html` и `speedup.html` (и PNG, если установлен kaleido).
