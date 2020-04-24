#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

using namespace std;

int num_vertices;
int num_edges;
int max_degree;

void printGraph(int n, int m[], int *adj_list) {
    for (int i = 0; i < n; ++i) {
        cout << "Node " << i << " || ";
        for (int j = 0; j < m[i]; ++j) {
            cout << adj_list[i * max_degree + j] << "    ";
        }
        cout << endl;
    }
}

void setGraph(int n, int m[], int *adj_list) {
    // Vertex 0
    adj_list[0 * max_degree + 0] = 1;
    adj_list[0 * max_degree + 1] = 3;

    // Vertex 1
    adj_list[1 * max_degree + 0] = 0;
    adj_list[1 * max_degree + 1] = 2;
    adj_list[1 * max_degree + 2] = 3;

    // Vertex 2
    adj_list[2 * max_degree + 0] = 1;
    adj_list[2 * max_degree + 1] = 3;

    // Vertex 3
    adj_list[3 * max_degree + 0] = 0;
    adj_list[3 * max_degree + 1] = 1;
    adj_list[3 * max_degree + 2] = 2;
}

__global__ void assign_colors_kernel(int num_conflicts, int *conflicts, int maxd, int *m, int *adj_list, int *colors,
                                     bool* forbidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < maxd + 1; ++j) {
        forbidden[i*(maxd+1) + j] = false;
    }

    int v = conflicts[i];

    for (int j = 0; j < m[v]; ++j) {
        int u = adj_list[v * maxd + j];
        if (colors[u] >= 0)
            forbidden[i * (maxd + 1) + colors[u]] = true;
    }

    for (int j = 0; j < maxd + 1; ++j) {
        if (forbidden[i * (maxd + 1) + j] == false) {
            colors[v] = j;
            break;
        }
    }
}

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int *adj_list, int *colors) {
    bool *forbidden = (bool *) malloc(num_conflicts * (maxd + 1) * sizeof(bool));

    int* device_conflicts;
    int* device_m;
    int* device_adj_list;
    int* device_colors;
    bool* device_forbidden;

    cudaMalloc((void **) &device_conflicts, num_conflicts * sizeof(int));
    cudaMalloc((void**) &device_m, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_adj_list, maxd * num_vertices * sizeof(int));
    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_forbidden, (maxd+1) * num_conflicts * sizeof(bool));

    cudaMemcpy(device_conflicts, conflicts, num_conflicts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_adj_list, adj_list, maxd * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_forbidden, forbidden, num_conflicts * (maxd + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    assign_colors_kernel <<<1, num_conflicts>>> (num_conflicts, device_conflicts, maxd, device_m, device_adj_list,
            device_colors, device_forbidden);
    cudaDeviceSynchronize();

    cudaMemcpy(conflicts, device_conflicts, num_conflicts * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(colors, device_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
}

void detect_conflicts(int num_conflicts, int *conflicts, int *m, int *adj_list, int *colors,
                      int *temp_num_conflicts, int *temp_conflicts) {
    for (int i = 0; i < num_conflicts; ++i) {
        int v = conflicts[i];
        for (int j = 0; j < m[v]; ++j) {
            int u = adj_list[v * max_degree + j];
            if (colors[u] == colors[v] && u < v) {
                temp_conflicts[*temp_num_conflicts] = u;
                *temp_num_conflicts = *temp_num_conflicts + 1;
                colors[u] = -u;
            }
        }
    }
}

void IPGC(int n, int m[], int maxd, int *adj_list) {
    int *colors = (int *) calloc(n, sizeof(int));
    int num_conflicts = n;
    int *conflicts = (int *) malloc(num_conflicts * sizeof(int));
    for (int i = 0; i < n; ++i) {
        conflicts[i] = i;
    }
    int temp_num_conflicts = 0;
    int *temp_conflicts = (int *) malloc(num_conflicts * sizeof(int));

    while (num_conflicts) {
        assign_colors(num_conflicts, conflicts, maxd, m, adj_list, colors);

        detect_conflicts(num_conflicts, conflicts, m, adj_list, colors, &temp_num_conflicts, temp_conflicts);

        // Swap
        num_conflicts = temp_num_conflicts;
        int *temp;
        temp = temp_conflicts;
        temp_conflicts = conflicts;
        conflicts = temp;
        temp_num_conflicts = 0;
    }

    for (int i = 0; i < n; ++i) {
        cout << "Color of node " << i << " : " << colors[i] << endl;
    }
}

int main() {
    int nvertices = 4;
    num_vertices = nvertices;
    int num_edges[] = {2, 3, 2, 3};
    max_degree = 3;
    int *adjacency_list = (int *) calloc(max_degree * nvertices, sizeof(int));

    setGraph(nvertices, num_edges, adjacency_list);
    printGraph(nvertices, num_edges, adjacency_list);
    IPGC(nvertices, num_edges, max_degree, adjacency_list);
}