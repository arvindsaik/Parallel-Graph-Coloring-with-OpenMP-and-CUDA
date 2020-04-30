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

int* device_conflicts;
int* device_m;
int* device_adj_list;
int* device_colors;
int* device_new_colors;
bool* device_forbidden;
int* device_temp_conflicts;

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

__global__ void upsweep(int *data, int N, int twod, int twod1) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (i + twod1 - 1 < N)
        data[i + twod1 - 1] += data[i + twod - 1];
}

__global__ void downsweep(int *data, int twod, int twod1) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    int t = data[i + twod - 1];
    data[i + twod - 1] = data[i + twod1 - 1];
    data[i + twod1 - 1] += t;
}

__global__ void findConflicts(int* conflicts, int* temp_conflicts) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (temp_conflicts[i] < temp_conflicts[i+1]) {
        conflicts[temp_conflicts[i]] = i;
    }
}

__global__ void detectConflictsKernel(int* conflicts, int* adj_list, int* temp_conflicts, int* colors, int* m, int
max_degree) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int v = conflicts[i];
    for (int j = 0; j < m[v]; ++j) {
        int u = adj_list[v * max_degree + j];
        if (colors[u] == colors[v] && u < v) {
            temp_conflicts[u] = 1;
            colors[u] = -u;
        }
    }
}

int nextPow2(int N)
{
    unsigned count = 0;

    if (N && !(N & (N - 1)))
        return N;

    while(N != 0)
    {   N>>= 1;
        count += 1;
    }

    return 1 << count;
}

void exclusive_scan(int *device_data, int length) {
    int orig_length = length;

    length = nextPow2(length);

    // compute number of blocks and threads per block

    // upsweep phase.
    for (int twod = 1; twod < length; twod *= 2) {
        int twod1 = twod * 2;
        const int threadsPerBlock = (512 > length / twod1) ? length / twod1 : 512;
        const int blocks = ((length / twod1) + threadsPerBlock - 1) / threadsPerBlock;
        upsweep << < blocks, threadsPerBlock >> > (device_data, orig_length, twod, twod1);
        cudaDeviceSynchronize();
    }

    // Setting the last element to zero
    cudaMemset(device_data + length - 1, 0, sizeof(int));

    // downsweep phase.
    for (int twod = length / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        const int threadsPerBlock = (512 > length / twod1) ? length / twod1 : 512;
        const int blocks = ((length / twod1) + threadsPerBlock - 1) / threadsPerBlock;
        downsweep << < blocks, threadsPerBlock >> > (device_data, twod, twod1);
        cudaDeviceSynchronize();
    }
}

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int *adj_list, int *colors) {
    cudaMemcpy(device_conflicts, conflicts, num_conflicts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_new_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    assign_colors_kernel <<<1, num_conflicts>>> (num_conflicts, device_conflicts, maxd, device_m, device_adj_list,
            device_colors, device_new_colors, device_forbidden);
    cudaDeviceSynchronize();

    cudaMemcpy(colors, device_new_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
}

void detect_conflicts(int num_conflicts, int *conflicts, int *m, int *adj_list, int *colors,
                      int *temp_num_conflicts, int *temp_conflicts) {
    cudaMemcpy(device_conflicts, conflicts, (num_conflicts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    detectConflictsKernel<<<1, num_conflicts>>> (device_conflicts, device_adj_list, device_temp_conflicts, device_colors,
            device_m, max_degree);
    cudaDeviceSynchronize();

    exclusive_scan(device_temp_conflicts, num_vertices + 1);

    cudaMemcpy(temp_num_conflicts, device_temp_conflicts + num_vertices, sizeof(int),
               cudaMemcpyDeviceToHost);

    findConflicts<<<1, num_vertices>>> (device_conflicts, device_temp_conflicts);
    cudaDeviceSynchronize();

    cudaMemcpy(temp_conflicts, device_conflicts, (num_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(colors, device_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
}

int* IPGC(int n, int m[], int maxd, int *adj_list) {
    int *colors = (int *) calloc(n, sizeof(int));
    int num_conflicts = n;
    int *conflicts = (int *) malloc((num_conflicts+1) * sizeof(int));
    for (int i = 0; i < n; ++i) {
        conflicts[i] = i;
    }
    int temp_num_conflicts = 0;
    int *temp_conflicts = (int *) malloc((num_conflicts+1) * sizeof(int));

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

    cudaMalloc((void **) &device_conflicts, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_m, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_adj_list, max_degree * num_vertices * sizeof(int));
    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_new_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_forbidden, (max_degree+1) * num_vertices * sizeof(bool));
    cudaMalloc((void**) &device_temp_conflicts, (num_vertices + 1) * sizeof(int));

    cudaMemcpy(device_adj_list, adjacency_list, max_degree * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, num_edges, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    int *colors = IPGC(nvertices, num_edges, max_degree, adjacency_list);
    cout << "Coloring done!" << endl;
    if (checker(nvertices, num_edges, colors, adjacency_list)) {
        cout << "CORRECT COLORING!!!" << endl;
    } else {
        cout << "INCORRECT COLORING!!!" << endl;
    }
}