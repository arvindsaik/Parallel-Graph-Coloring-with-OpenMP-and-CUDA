#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

using namespace std;

int max_degree;
int num_vertices;
int num_edges;
int *edges;

int* device_colors;
bool* device_vforbidden;
int* device_edges;
int* device_temp_colors;
bool* device_is_conflict;

__global__ void assign_colors_kernel(int *colors, bool* vforbidden, int max_degree, int num_vertices) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= num_vertices) return;
    if (colors[i] != -1)
        return;

    for (int j = 0; j < max_degree + 1; ++j) {
        if (!vforbidden[i * (max_degree + 1) + j]) {
            colors[i] = j;
            break;
        }
    }
}

__global__ void detect_conflicts_kernel(int* edges, int* colors, int* temp_colors, bool* vforbidden,
        bool* is_conflict, int max_degree, int num_edges) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= num_edges) return;

    int smaller_vertex, bigger_vertex;
    if (edges[2*i] > edges[2*i+1]) {
        bigger_vertex = edges[2*i];
        smaller_vertex = edges[2*i+1];
    } else {
        bigger_vertex = edges[2*i+1];
        smaller_vertex = edges[2*i];
    }
    if (colors[smaller_vertex] == colors[bigger_vertex]) {
        temp_colors[smaller_vertex] = -1;
        if (*is_conflict == false)
            *is_conflict = true;
    }
    vforbidden[smaller_vertex * (max_degree+1) + colors[bigger_vertex]] = true;
}

void assign_colors() {
    assign_colors_kernel<<<(num_vertices+1023)/1024, 1024>>>(device_colors, device_vforbidden, max_degree,
            num_vertices);
    cudaDeviceSynchronize();
}

bool detect_conflicts() {
    bool is_conflict = false;
    cudaMemset(device_is_conflict, false, sizeof(bool));

    detect_conflicts_kernel<<<(num_edges + 1023)/1024, 1024>>>(device_edges, device_colors, device_temp_colors,
            device_vforbidden,
            device_is_conflict, max_degree, num_edges);
    cudaDeviceSynchronize();

    cudaMemcpy(&is_conflict, device_is_conflict, sizeof(bool), cudaMemcpyDeviceToHost);
    return is_conflict;
}

int *IPGC() {
    int *colors = (int *) malloc(num_vertices * sizeof(int));
    cudaMemset(device_colors, -1, num_vertices * sizeof(int));

    long iter = 0;
    int is_conflict = true;

    while (is_conflict) {
        cout << __LINE__ << endl;
        fflush(stdin);
        fflush(stdout);

        iter++;
        assign_colors();

        cudaMemset(device_vforbidden, 0, num_vertices * (max_degree + 1) * sizeof(bool));
        cudaMemcpy(device_temp_colors, device_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToDevice);

        is_conflict = detect_conflicts();

        cudaMemcpy(device_colors, device_temp_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(colors, device_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Iteration " << iter << endl;
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Color of node " << i << " : " << colors[i] << endl;
    }
    fflush(stdin);
    fflush(stdout);
    return colors;
}

bool checker(int *colors) {
    bool passed = true;
    for (int i = 0; i < num_edges; ++i) {
        if (colors[edges[2*i]] == colors[edges[2*i+1]] ||
        colors[edges[2*i]] < 0 || colors[edges[2*i+1]] < 0 ||
        colors[edges[2*i]] > max_degree + 1 || colors[edges[2*i+1]] > max_degree + 1) {
            passed = false;
        }
    }
    return passed;
}

int main(int argc, char *argv[]) {
    char *filename = argv[1];

    cout << filename << endl;
    ifstream fin(filename);
    fin >> max_degree;
    fin >> (num_vertices);
    fin >> (num_edges);
    cout << max_degree << " : " << num_vertices << " : " << num_edges << endl;
    fflush(stdin);
    fflush(stdout);
    edges = (int *) malloc(2 * num_edges * sizeof(int));

    for (int i = 0; i < num_edges; ++i) {
        fin >> edges[2*i];
        fin >> edges[2*i+1];
        fflush(stdin);
        fflush(stdout);
    }
    fin.close();

    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_vforbidden, num_vertices * (max_degree + 1) * sizeof(bool));
    cudaMalloc((void**) &device_edges, 2 * num_edges * sizeof(int));
    cudaMalloc((void**) &device_temp_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_is_conflict, sizeof(bool));

    cudaMemcpy(device_edges, edges, 2 * num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int *colors = IPGC();

    cout << "Coloring done!" << endl;
    if (checker(colors)) {
        cout << "CORRECT COLORING!!!" << endl;
    } else {
        cout << "INCORRECT COLORING!!!" << endl;
    }
}