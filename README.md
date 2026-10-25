# Практическая работа №1: CUDA Vector

> Практическая работа №2 (умножение матриц на CUDA) — в папке [`work2/`](work2/).


Проект самостоятельный и не содержит Docker или Dev Container файлов. Его
можно открыть обычным CMake-проектом, а затем открыть той же папкой внутри
CUDA-контейнера.

## Сборка

Из каталога проекта:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
```

Если `native` не поддерживается вашей версией CMake, укажите архитектуру
видеокарты вручную, например `-DCMAKE_CUDA_ARCHITECTURES=86`.

## Тесты

```bash
ctest --test-dir build --output-on-failure
```

## Бенчмарки и графики

```bash
./build/cuda_work1_benchmarks \
  --benchmark_out=build/benchmarks.json \
  --benchmark_out_format=json
python3 -m pip install plotly kaleido
python3 scripts/plot_benchmarks.py build/benchmarks.json build/plots
```

Графики появятся в `build/plots`.
