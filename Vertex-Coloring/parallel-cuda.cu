#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

using namespace std;

int num_vertices;
int num_edges;

void printGraph(int n, int m[], int **adj_list) {
    for (int i = 0; i < n; ++i) {
        cout << "Node " << i << " || ";
        for (int j = 0; j < m[i]; ++j) {
            cout << adj_list[i][j] << "    ";
        }
        cout << endl;
    }
}

void setGraph(int n, int m[], int **adj_list) {
    // Vertex 0
    adj_list[0][0] = 1;
    adj_list[0][1] = 3;

    // Vertex 1
    adj_list[1][0] = 0;
    adj_list[1][1] = 2;
    adj_list[1][2] = 3;

    // Vertex 2
    adj_list[2][0] = 1;
    adj_list[2][1] = 3;

    // Vertex 3
    adj_list[3][0] = 0;
    adj_list[3][1] = 1;
    adj_list[3][2] = 2;
}

__global__ assign_colors_kernel(int num_conflicts, int *conflicts, int maxd, int *m, int **adj_list, int *colors) {
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

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int **adj_list, int *colors) {
    bool **forbidden = (bool **) malloc(num_conflicts * sizeof(bool *));
    for (int i = 0; i < num_conflicts; ++i) {
        forbidden[i] = (bool *) malloc((maxd + 1) * sizeof(bool));
    }

    int* device_conflicts;
    int* device_m;
    int* device_adj_list;
    int* device_colors;
    bool* device_forbidden;

    cudaMalloc((void **) &device_conflicts, num_conflicts * sizeof(int));
    cudaMalloc((void**) &m, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_adj_list, maxd * num_vertices * sizeof(int));
    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_forbidden, (maxd+1) * num_conflicts * sizeof(bool));

    cudaMemcpy(device_conflicts, conflicts, * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < num_conflicts; ++i) {

    }
}

void detect_conflicts(int num_conflicts, int *conflicts, int *m, int **adj_list, int *colors,
                      int *temp_num_conflicts, int *temp_conflicts) {
#if OMP
    #pragma omp parallel num_threads(4)
		#pragma omp for
#endif
    for (int i = 0; i < num_conflicts; ++i) {
        int v = conflicts[i];
        for (int j = 0; j < m[v]; ++j) {
            int u = adj_list[v][j];
            if (colors[u] == colors[v] && u < v) {
#if OMP
#pragma omp critical
#endif
                temp_conflicts[*temp_num_conflicts] = u;
                *temp_num_conflicts = *temp_num_conflicts + 1;
                colors[u] = -u;
            }
        }
    }
}

void IPGC(int n, int m[], int maxd, int **adj_list) {
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
    int max_degree = 3;
    int **adjacency_list = (int **) malloc(nvertices * sizeof(int *));
    for (int i = 0; i < nvertices; ++i) {
        adjacency_list[i] = (int *) malloc(max_degree * sizeof(int));
    }

    setGraph(nvertices, num_edges, adjacency_list);
    printGraph(nvertices, num_edges, adjacency_list);
    IPGC(nvertices, num_edges, max_degree, adjacency_list);
}