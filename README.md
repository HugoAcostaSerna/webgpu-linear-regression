# Implementación Vectorizada de Regresión Lineal en WebGPU (WGSL)

Implementación nativa de Descenso de Gradiente en WebGPU utilizando Compute Shaders en WGSL y evaluación comparativa frente a TensorFlow.js y PyTorch/TensorFlow.

## Requisitos Previos
- Navegador compatible con WebGPU (Google Chrome 113+ o Microsoft Edge 113+) con aceleración por hardware activada.
- Node.js v18+ y npm.
- Python 3.10+ (para los benchmarks de Baseline 2).

## Instalación y Ejecución (WebGPU y TF.js)
```bash
npm install
npm run dev