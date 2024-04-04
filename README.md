# Taller de CUDA

Para compilar compilar y ejecutar los archivos usando CUDA:

1. **Suma vectorial:**
```
nvcc vecadd.cu -o vecadd && ./vecadd
```
2. **Multiplicación de matrices:**
```
nvcc mat_mul.cu -o mat_mul && ./mat_mul
```
3. **Filtro de escala de grises:**
```
cd grayscale_filter
make
```
