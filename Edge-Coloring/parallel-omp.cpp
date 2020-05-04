#include <bits/stdc++.h>
#include <omp.h>
//#include <ctime>
#include "cycletimer.h"


using namespace std;
#define NUM_THREADS 12
#define BILLION  1000000000.0

double assign_colors_time = 0;
double detect_conflicts_time = 0;
double total_time = 0;
int iterations = 0;

void assign_colors(int n, int maxd, int *colors, bool **vforbidden) {
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (int i = 0; i < n; ++i) {
		if (colors[i] != -1) continue;
		for (int j = 0; j < maxd + 1; ++j) {
			if (vforbidden[i][j] == false) {
				colors[i] = j;
				break;
			}
		}
	}
}

bool detect_conflicts(int num_edges, int **edges, int *colors, int *temp_colors, bool **vforbidden) {
	bool is_conflict = false;
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (int  i = 0; i < num_edges; ++i) {
		int smaller_vertex, bigger_vertex;
		if (edges[i][0] > edges[i][1]) {
			bigger_vertex = edges[i][0];
			smaller_vertex = edges[i][1];
		} else {
			bigger_vertex = edges[i][1];
			smaller_vertex = edges[i][0];
		}
		if (colors[smaller_vertex] == colors[bigger_vertex]) {
			temp_colors[smaller_vertex] = -1;
			if (!is_conflict)
				is_conflict = true;
		}
		if (!vforbidden[smaller_vertex][colors[bigger_vertex]])
			vforbidden[smaller_vertex][colors[bigger_vertex]] = true;
	}
	return is_conflict;
}

int *IPGC(int n, int num_edges, int maxd, int **edges) {
	int *colors = (int *) malloc(n * sizeof(int));
	int *temp_colors = (int *) malloc(n * sizeof(int));
	for (int i = 0; i < n; ++i) {
		colors[i] = -1;
		temp_colors[i] = -1;
	}
	bool **vforbidden = (bool **) malloc(n * sizeof(bool *));
	for (int i = 0; i < n; ++i) {
		vforbidden[i] = (bool *) calloc(maxd + 1, sizeof(bool));
	}
	int iter = 0;
	int is_conflict = true;

    struct timespec start, end, start1, end1;

    clock_gettime(CLOCK_REALTIME, &start);

	while (is_conflict) {
        iterations++;
        clock_gettime(CLOCK_REALTIME, &start1);
		assign_colors(n, maxd, colors, vforbidden);
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
		for (int i = 0; i < n; ++i) {
			memset(vforbidden[i], 0, sizeof(vforbidden[i]));
			temp_colors[i] = colors[i];
		}
        clock_gettime(CLOCK_REALTIME, &end1);
        assign_colors_time += (end1.tv_sec - start1.tv_sec) +
                              (end1.tv_nsec - start1.tv_nsec) / BILLION;

        clock_gettime(CLOCK_REALTIME, &start1);
        is_conflict = detect_conflicts(num_edges, edges, colors, temp_colors, vforbidden);

#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
		for (int i = 0; i < n; ++i) {
			colors[i] = temp_colors[i];
		}
        clock_gettime(CLOCK_REALTIME, &end1);
        detect_conflicts_time += (end1.tv_sec - start1.tv_sec) +
                                 (end1.tv_nsec - start1.tv_nsec) / BILLION;
	}
//	for (int i = 0; i < n; ++i) {
//		cout << "Color of node " << i << " : " << colors[i] << endl;
//	}
    clock_gettime(CLOCK_REALTIME, &end);
    total_time = (end.tv_sec - start.tv_sec) +
                 (end.tv_nsec - start.tv_nsec) / BILLION;

	return colors;
}

bool checker(int num_edges, int **edges, int *colors, int maxd) {
	bool passed = true;
	for (int i = 0; i < num_edges; ++i) {
		if (colors[edges[i][0]] < 0 || colors[edges[i][0]] > maxd) {
			passed = false;
		}
		if (colors[edges[i][1]] < 0 || colors[edges[i][1]] > maxd) {
			passed = false;
		}
		if (colors[edges[i][0]] == colors[edges[i][1]]) {
			passed = false;
		}
	}
	return passed;
}

int main(int argc, char *argv[]) {
	int nvertices;
	int num_edges;
	int max_degree;
	char *filename = argv[1];

	// Read graph from file
	cout << filename << endl;
	ifstream fin(filename);
	fin >> max_degree;
	fin >> (nvertices);
	fin >> (num_edges);
	cout << max_degree << " : " << nvertices << " : " << num_edges << endl;
	fflush(stdin);
	fflush(stdout);
	int **edges = (int **) malloc(num_edges * sizeof(int *));
	for (int i = 0; i < num_edges; ++i) {
		edges[i] = (int *) malloc(2 * sizeof(int));
	}

	for (int i = 0; i < num_edges; ++i) {
		fin >> edges[i][0];
		fin >> edges[i][1];
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

	// Perform coloring
	int *colors = IPGC(nvertices, num_edges, max_degree, edges);

    cout << "Total time for coloring : " << total_time * 1000 << " ms" << endl;
    cout << "Time taken for Assign Colors : " << assign_colors_time * 1000 << " ms" << endl;
    cout << "Time taken for Detect Conflicts : " << detect_conflicts_time * 1000 << " ms" << endl;
    cout << "Iterations taken to converge : " << iterations << endl;
    int max_color = -1;
    for (int i = 0; i < nvertices; ++i) {
        max_color = max(max_color, colors[i]);
    }
    cout << "Colors used in the coloring : " << max_color + 1 << endl;

	// Call checker
	if (checker(num_edges, edges, colors, max_degree)) {
		cout << "CORRECT COLORING!!!" << endl;
	} else {
		cout << "INCORRECT COLORING!!!" << endl;
	}
}