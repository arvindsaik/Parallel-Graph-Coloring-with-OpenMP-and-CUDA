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
int *num_edges_per_vertex;
int *adjacency_list;

int* device_m;
int* device_adj_list;
int* device_conflicts;
int* device_temp_conflicts;
int* device_colors;
int* device_new_colors;
bool* device_forbidden;

#define BILLION  1000000000.0

double assign_colors_time = 0;
double detect_conflicts_time = 0;
double total_time = 0;
double memory_ops_time = 0;
int iterations = 0;
struct timespec m_start, m_end;

void printGraph(int n, int m[], int *adj_list) {
    for (int i = 0; i < n; ++i) {
        cout << "Node " << i << " || ";
        for (int j = 0; j < m[i]; ++j) {
            cout << adj_list[i * max_degree + j] << "    ";
        }
        cout << endl;
    }
}

__global__ void assign_init_values(int* conflicts, int num_vertices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    conflicts[i] = i;
}

__global__ void assign_colors_kernel(int num_conflicts, int *conflicts, int maxd, int *m, int *adj_list, int *colors,
                                     int* new_colors, bool* forbidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_conflicts) return;

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

__global__ void findConflicts(int* conflicts, int* temp_conflicts, int num_vertices) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_vertices) return;
    if (temp_conflicts[i] < temp_conflicts[i+1]) {
        conflicts[temp_conflicts[i]] = i;
    }
}

__global__ void detectConflictsKernel(int* conflicts, int* adj_list, int* temp_conflicts, int* colors, int* m, int
max_degree, int num_vertices) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_vertices) return;

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

void assign_colors(int num_conflicts) {
    clock_gettime(CLOCK_REALTIME, &m_start);
    cudaMemcpy(device_colors, device_new_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToDevice);
    clock_gettime(CLOCK_REALTIME, &m_end);

    memory_ops_time += (m_end.tv_sec - m_start.tv_sec) +
                       (m_end.tv_nsec - m_start.tv_nsec) / BILLION;

    assign_colors_kernel <<<(num_conflicts+1023)/1024, 1024>>> (num_conflicts, device_conflicts, max_degree, device_m,
            device_adj_list,
            device_colors, device_new_colors, device_forbidden);
    cudaDeviceSynchronize();
}

void detect_conflicts(int num_conflicts, int *temp_num_conflicts) {
    clock_gettime(CLOCK_REALTIME, &m_start);
    cudaMemset((void*) device_temp_conflicts, 0, (num_vertices+1) * sizeof(int));
    clock_gettime(CLOCK_REALTIME, &m_end);
    memory_ops_time += (m_end.tv_sec - m_start.tv_sec) +
                       (m_end.tv_nsec - m_start.tv_nsec) / BILLION;

    detectConflictsKernel<<<(num_conflicts+1023)/1024, 1024>>> (device_conflicts, device_adj_list, device_temp_conflicts,
            device_new_colors, device_m, max_degree, num_conflicts);
    cudaDeviceSynchronize();

    exclusive_scan(device_temp_conflicts, num_vertices + 1);

    cudaMemcpy(temp_num_conflicts, device_temp_conflicts + num_vertices, sizeof(int),
               cudaMemcpyDeviceToHost);

    findConflicts<<<(num_vertices+1023)/1024, 1024>>> (device_conflicts, device_temp_conflicts, num_vertices);
    cudaDeviceSynchronize();
}

int* IPGC() {
    int *colors = (int *) calloc(num_vertices, sizeof(int));
    int num_conflicts = num_vertices;

    assign_init_values<<<(num_vertices+1023)/1024, 1024>>>(device_conflicts, num_vertices);

    int temp_num_conflicts = 0;

    struct timespec start, end, start1, end1;

    clock_gettime(CLOCK_REALTIME, &start);
    while (num_conflicts) {
        iterations++;

        clock_gettime(CLOCK_REALTIME, &start1);
        assign_colors(num_conflicts);
        clock_gettime(CLOCK_REALTIME, &end1);
        assign_colors_time += (end1.tv_sec - start1.tv_sec) +
                              (end1.tv_nsec - start1.tv_nsec) / BILLION;

        clock_gettime(CLOCK_REALTIME, &start1);
        detect_conflicts(num_conflicts, &temp_num_conflicts);
        clock_gettime(CLOCK_REALTIME, &end1);
        detect_conflicts_time += (end1.tv_sec - start1.tv_sec) +
                                 (end1.tv_nsec - start1.tv_nsec) / BILLION;

        num_conflicts = temp_num_conflicts;
        temp_num_conflicts = 0;
    }
    clock_gettime(CLOCK_REALTIME, &end);
    total_time += (end.tv_sec - start.tv_sec) +
                 (end.tv_nsec - start.tv_nsec) / BILLION;

    cudaMemcpy(colors, device_new_colors, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < num_vertices; ++i) {
//        cout << "Color of node " << i << " : " << colors[i] << endl;
//    }
    fflush(stdin);
    fflush(stdout);
    return colors;
}

bool checker(int nvertices, int *num_edges, int *colors, int *adjacency_list) {
    bool passed = true;
    for (int i = 0; i < nvertices; ++i) {
        for (int j = 0; j < num_edges[i]; ++j) {
            if (colors[i] == colors[adjacency_list[i * max_degree + j]] || colors[i] < 0 || colors[i] > max_degree +
                                                                                                        1) {
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
    char *filename = argv[1];

    ifstream fin(filename);
    fin >> (max_degree);
    fin >> (num_vertices);

    adjacency_list = (int *) malloc(num_vertices * (max_degree) * sizeof(int));
    num_edges_per_vertex = (int *) malloc(num_vertices * sizeof(int ));

    for (int i = 0; i < num_vertices; ++i) {
        fin >> num_edges_per_vertex[i];
        for (int j = 0; j < num_edges_per_vertex[i]; ++j) {
            fin >> adjacency_list[i * max_degree + j];
        }
    }
    fin.close();

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMalloc((void**) &device_m, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_adj_list, max_degree * num_vertices * sizeof(int));
    cudaMalloc((void**) &device_temp_conflicts, nextPow2(num_vertices + 1) * sizeof(int));
    cudaMalloc((void**) &device_conflicts, (num_vertices + 1) * sizeof(int));
    cudaMalloc((void**) &device_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_new_colors, num_vertices * sizeof(int));
    cudaMalloc((void**) &device_forbidden, (max_degree+1) * num_vertices * sizeof(bool));

    cudaMemcpy(device_adj_list, adjacency_list, max_degree * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, num_edges_per_vertex, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_REALTIME, &end);
    total_time += (end.tv_sec - start.tv_sec) +
                 (end.tv_nsec - start.tv_nsec) / BILLION;
    memory_ops_time += (end.tv_sec - start.tv_sec) +
                       (end.tv_nsec - start.tv_nsec) / BILLION;

//	printGraph(nvertices, num_edges, adjacency_list);
    int *colors = IPGC();
    cout << "Total time for coloring : " << total_time * 1000 << " ms" << endl;
    cout << "Time taken for Assign Colors : " << assign_colors_time * 1000 << " ms" << endl;
    cout << "Time taken for Detect Conflicts : " << detect_conflicts_time * 1000 << " ms" << endl;
    cout << "Time taken for Memory operations: " << memory_ops_time * 1000 << " ms" << endl;
    cout << "Iterations taken to converge : " << iterations << endl;
    int max_color = -1;
    for (int i = 0; i < num_vertices; ++i) {
        max_color = max(max_color, colors[i]);
    }
    cout << "Colors used in the coloring : " << max_color + 1 << endl;

    if (checker(num_vertices, num_edges_per_vertex, colors, adjacency_list)) {
        cout << "CORRECT COLORING!!!" << endl;
    } else {
        cout << "INCORRECT COLORING!!!" << endl;
    }
}