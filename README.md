# Ultrasound HPC
## Procesamiento de Ultrasonido Médico mediante Computación de Alto Desempeño

Proyecto final de la asignatura **Temas Selectos de Cómputo de Alto Desempeño (HPC)**, enfocado en la aceleración de técnicas de procesamiento de imágenes médicas utilizando arquitecturas heterogéneas.

El proyecto explora y compara tres enfoques de procesamiento:

- CPU secuencial
- CPU multihilo mediante OpenMP
- GPU mediante OpenCL

A través de un pipeline completo de procesamiento de video médico, se evalúan tanto la calidad visual de las imágenes como el rendimiento computacional de cada arquitectura.

---

# Descripción General

Las imágenes de ultrasonido suelen verse afectadas por **ruido speckle**, un fenómeno inherente al proceso de adquisición que degrada la calidad visual y dificulta la identificación de estructuras anatómicas.

Este proyecto implementa diversas técnicas de mejora de imagen con el objetivo de reducir dicho ruido, preservar bordes importantes y analizar el impacto de la paralelización sobre el tiempo de procesamiento.

Los filtros implementados incluyen:

- Filtro Bilateral
- Sharpen (Realce de Bordes)

---

# Objetivos

- Reducir el ruido speckle presente en imágenes de ultrasonido.
- Mejorar la calidad visual preservando estructuras anatómicas relevantes.
- Comparar arquitecturas secuenciales y paralelas.
- Implementar algoritmos sobre CPU y GPU.
- Medir el desempeño mediante métricas de rendimiento.
- Evaluar objetivamente la calidad de las imágenes procesadas.

---

# Arquitecturas Implementadas

## Procesamiento Secuencial

Implementación en Python utilizando OpenCV.

Características:

- Procesamiento frame por frame.
- Un único hilo de ejecución.
- Utilizado como línea base para calcular speedup.

---

## Procesamiento Paralelo con OpenMP

Implementación en C++.

Estrategia de paralelización:

- Un frame por hilo.
- Memoria compartida.
- Paralelismo a nivel de tareas.
- Bajo costo de sincronización.

---

## Procesamiento Paralelo con OpenCL

Implementación mediante PyOpenCL.

Estrategia de paralelización:

- Un work-item por píxel.
- Ejecución masiva sobre GPU.
- Implementación de kernels para filtros de imagen.
- Evaluación del impacto del overhead de transferencia y ejecución.

---

# Pipeline de Procesamiento

```text
Video de Ultrasonido
          │
          ▼
Extracción de Frames
          │
          ▼
Procesamiento de Imagen
(Bilateral / Sharpen )
          │
          ▼
Benchmark de Rendimiento
          │
          ▼
Reconstrucción de Video
          │
          ▼
Evaluación de Calidad
```

---

# Métricas de Calidad de Imagen

Para evaluar objetivamente los resultados se calcularon:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SNR (Signal-to-Noise Ratio)
- Entropía
- Error RMS

Estas métricas permiten cuantificar la conservación de información y la mejora visual obtenida por cada técnica de procesamiento.

---

# Métricas de Rendimiento

Para cada arquitectura se midieron:

- Tiempo total de ejecución
- Tiempo promedio por frame
- Frames procesados por segundo (FPS)
- Speedup

Estas mediciones permiten comparar la eficiencia de las distintas estrategias de paralelización.

---

# Estructura del Proyecto

```text
ultrasound_hpc/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── metrics/
│
├── src/
│   ├── filters/
│   ├── preprocessing/
│   ├── parallel/
│   ├── metrics/
│   ├── io/
│   ├── visualization/
│   └── utils/
│
├── main.py
├── main_benchmark_cpu.py
├── main_benchmark_gpu.py
├── requirements.txt
└── README.md
```

---

# Tecnologías Utilizadas

## Lenguajes

- Python
- C++

## Librerías

- OpenCV
- NumPy
- PyOpenCL
- OpenMP

## Plataformas

- Windows 11
- WSL Ubuntu
- GPU AMD Integrada

---

# Resultados Principales

Los experimentos realizados permitieron observar que:

- OpenMP obtuvo el mejor desempeño en filtros de baja complejidad computacional.
- OpenCL mostró ventajas cuando la carga de trabajo aumentó.
- La aceleración mediante GPU no garantiza mejoras de rendimiento en todos los escenarios.
- El costo asociado al lanzamiento de kernels y la transferencia de memoria influye significativamente en el rendimiento final.
- La relación entre carga computacional y overhead es un factor crítico en aplicaciones HPC.

Los resultados obtenidos coinciden con conceptos fundamentales de computación paralela como la **Ley de Amdahl**, granularidad de tareas y balance entre cómputo y comunicación.

---

# Contribuciones Académicas

Este proyecto integra conceptos de:

- Procesamiento Digital de Imágenes
- Computación Paralela
- Computación de Alto Desempeño (HPC)
- Programación GPU con OpenCL
- Paralelización con OpenMP
- Benchmarking de Arquitecturas
- Procesamiento de Imágenes Médicas

Asimismo, demuestra la aplicación práctica de técnicas HPC sobre un problema real relacionado con el análisis de imágenes de ultrasonido.

---

# Autor

**Johnny Michael Galicia Orihuela**

Licenciatura en Física

Proyecto Final — Temas Selectos de Cómputo de Alto Desempeño

Universidad Nacional Autónoma de México UNAM 

2026

# Ultrasound HPC
## High-Performance Medical Ultrasound Processing using OpenCV, OpenMP and OpenCL

Final project for the High Performance Computing (HPC) course.

This project explores the acceleration of medical ultrasound image processing pipelines using heterogeneous computing architectures, including:

- Sequential CPU processing
- Multi-threaded CPU processing with OpenMP
- GPU acceleration with OpenCL

The study evaluates image quality and computational performance through a complete video-processing workflow.

---

# Project Overview

Medical ultrasound images are heavily affected by speckle noise, which degrades visual quality and complicates diagnostic interpretation.

This project implements and evaluates several enhancement techniques:

- Bilateral Filtering
- Image Sharpening

The objective is to improve image quality while analyzing the computational benefits and limitations of parallel architectures.

---

# Research Objectives

- Reduce speckle noise in ultrasound images.
- Improve visual quality while preserving anatomical structures.
- Compare sequential, multi-threaded, and GPU-based implementations.
- Measure computational performance using benchmarking metrics.
- Evaluate image quality using objective metrics.

---

# Implemented Architectures

## Sequential CPU

Python + OpenCV implementation.

Processing performed frame-by-frame using a single execution thread.

---

## OpenMP Multi-threading

C++ implementation.

Parallelization strategy:

- One frame per thread
- Shared-memory architecture
- Low synchronization overhead

---

## OpenCL GPU

PyOpenCL implementation.

Parallelization strategy:

- One work-item per pixel
- Bilateral filtering kernel
- Sharpen convolution kernel
- AMD integrated GPU execution

---

# Processing Pipeline

```text
Input Ultrasound Video
          │
          ▼
Frame Extraction
          │
          ▼
Image Processing
(Bilateral / Sharpen)
          │
          ▼
Benchmarking
          │
          ▼
Video Reconstruction
          │
          ▼
Quality Evaluation
```

---

# Image Quality Metrics

The following metrics were computed:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SNR (Signal-to-Noise Ratio)
- Entropy
- RMS Error

---

# Performance Metrics

For each architecture:

- Total execution time
- Time per frame
- Frames per second (FPS)
- Speedup

---

# Project Structure

```text
ultrasound_hpc/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── metrics/
│
├── src/
│   ├── filters/
│   ├── preprocessing/
│   ├── parallel/
│   ├── metrics/
│   ├── io/
│   ├── visualization/
│   └── utils/
│
├── main.py
├── main_benchmark_cpu.py
├── main_benchmark_gpu.py
├── requirements.txt
└── README.md
```

---

# Technologies

## Languages

- Python
- C++

## Libraries

- OpenCV
- NumPy
- PyOpenCL
- OpenMP

## Platforms

- Windows 11
- WSL Ubuntu
- AMD Integrated GPU

---

# Main Results

The experimental results showed that:

- OpenMP achieved the best performance for lightweight filters.
- OpenCL became advantageous for computationally intensive workloads.
- GPU acceleration is not always faster due to memory-transfer and kernel-launch overhead.
- The relationship between computation cost and parallelization overhead plays a critical role in overall performance.

These findings are consistent with classical HPC concepts such as Amdahl's Law and task granularity.

---

# Academic Contributions

This project demonstrates:

- Practical GPU programming with OpenCL
- Shared-memory parallelization with OpenMP
- Medical image enhancement techniques
- Performance benchmarking methodologies
- Comparative analysis of heterogeneous computing architectures

---

# Author

Johnny Michael Galicia Orihuela

Bachelor's Degree in Physics

High Performance Computing – Final Project

Universidad Nacional Autónoma de México UNAM 

2026
