#include <immintrin.h> // Para AVX intrinsics
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//c++ -O2 -mavx -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) avx_intrinsics.cpp -o avx_intr$(python3-config --extension-suffix)

namespace py = pybind11;

// Función para sumar dos vectores usando AVX
std::vector<float> add_vectors_avx(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t size = a.size();
    std::vector<float> result(size);

    size_t i = 0;

    // Proceso en bloques de 8 elementos con AVX
    for (; i + 8 <= size; i += 8) {
        // Cargar datos de los vectores en registros AVX
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        // Realizar la suma
        __m256 vr = _mm256_add_ps(va, vb);

        // Almacenar el resultado en el vector de salida
        _mm256_storeu_ps(&result[i], vr);
    }

    // Procesar cualquier elemento restante (no múltiplo de 8)
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// Vincular la función usando pybind11
PYBIND11_MODULE(avx_intr, m) {
    m.doc() = "AVX Intrinsics example with pybind11";
    m.def("add_vectors_avx", &add_vectors_avx, "Add two vectors using AVX intrinsics");
}
