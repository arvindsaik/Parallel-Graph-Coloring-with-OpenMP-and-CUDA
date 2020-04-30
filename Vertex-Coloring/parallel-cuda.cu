#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

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
        int* new_colors, bool* forbidden) {
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
        if (!forbidden[i * (maxd + 1) + j]) {
            new_colors[v] = j;
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
    int* device_new_colors;
    bool* device_forbidden;

    cudaMalloc((void **) &device_conflicts, num_conflicts * sizeof(int));
    cudaMalloc((void**) &device_m, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_adj_list, maxd * num_vertices * sizeof(int));
    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_new_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_forbidden, (maxd+1) * num_conflicts * sizeof(bool));

    cudaMemcpy(device_conflicts, conflicts, num_conflicts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_adj_list, adj_list, maxd * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_new_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    assign_colors_kernel <<<1, num_conflicts>>> (num_conflicts, device_conflicts, maxd, device_m, device_adj_list,
            device_colors, device_new_colors, device_forbidden);
    cudaDeviceSynchronize();

    cudaMemcpy(colors, device_new_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
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

int* IPGC(int n, int m[], int maxd, int *adj_list) {
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

    fflush(stdin);
    fflush(stdout);
    return colors;
}

bool checker(int nvertices, int *num_edges, int *colors, int *adjacency_list) {
    bool passed = true;
    for (int i = 0; i < nvertices; ++i) {
        for (int j = 0; j < num_edges[i]; ++j) {
            if (colors[i] == colors[adjacency_list[i * max_degree + j]]) {
                passed = false;
                cout << "Failed coloring between nodes : " << i << " -- " << adjacency_list[i * max_degree + j];
                fflush(stdin);
                fflush(stdout);
                break;
            }
        }
    }
    return passed;
}

int main(int argc, char *argv[]) {
    int nvertices;
    int *num_edges;
    int *adjacency_list;
    char *filename = argv[1];

    ifstream fin(filename);
    fin >> (max_degree);
    fin >> (nvertices);
    num_vertices = nvertices;
    fflush(stdin);
    fflush(stdout);

    adjacency_list = (int *) malloc(nvertices * (max_degree) * sizeof(int));
    num_edges = (int *) malloc(nvertices * sizeof(int ));

    for (int i = 0; i < nvertices; ++i) {
        fin >> num_edges[i];
        fflush(stdin);
        fflush(stdout);
        for (int j = 0; j < num_edges[i]; ++j) {
            fin >> adjacency_list[i * max_degree + j];
            fflush(stdin);
            fflush(stdout);
        }
        fflush(stdin);
        fflush(stdout);
    }
    fin.close();

//	printGraph(nvertices, num_edges, adjacency_list);
    int *colors = IPGC(nvertices, num_edges, max_degree, adjacency_list);
    cout << "Coloring done!" << endl;
    if (checker(nvertices, num_edges, colors, adjacency_list)) {
        cout << "CORRECT COLORING!!!" << endl;
    } else {
        cout << "INCORRECT COLORING!!!" << endl;
    }
}