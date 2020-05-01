#include <bits/stdc++.h>
#include <omp.h>


using namespace std;
#define NUM_THREADS 10

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int **adj_list, int *colors) {
	bool **forbidden = (bool **) malloc(num_conflicts * sizeof(bool *));
	for (int i = 0; i < num_conflicts; ++i) {
		forbidden[i] = (bool *) malloc(maxd * sizeof(bool));
	}
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (int i = 0; i < num_conflicts; ++i) {
		for (int j = 0; j < maxd + 1; ++j) {
			forbidden[i][j] = false;
		}
		int v = conflicts[i];
		for (int j = 0; j < m[v]; ++j) {
			int u = adj_list[v][j];
			if (colors[u] >= 0)
				forbidden[i][colors[u]] = true;
		}
		for (int j = 0; j < maxd + 1; ++j) {
			if (forbidden[i][j] == false) {
				colors[v] = j;
				break;
			}
		}
	}
}

void detect_conflicts(int num_conflicts, int *conflicts, int *m, int **adj_list, int *colors,
                      int *temp_num_conflicts, int *temp_conflicts) {
	#if OMP
		#pragma omp parallel num_threads(NUM_THREADS)
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
				{
					temp_conflicts[*temp_num_conflicts] = u;
					*temp_num_conflicts = *temp_num_conflicts + 1;
					colors[u] = -u;
				}
			}
		}
	}
}

int * IPGC(int n, int m[], int maxd, int **adj_list) {
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

	fflush(stdin);
	fflush(stdout);
	return colors;
}

bool checker(int nvertices, int maxd, int *num_edges, int *colors, int **adjacency_list) {
	bool passed = true;
	for (int i = 0; i < nvertices; ++i) {
		if (colors[i] < 0 || colors[i] > maxd) {
			passed = false;
			cout << "Wrong coloring of vertex : " << i << endl;
			break;
		}
		for (int j = 0; j < num_edges[i]; ++j) {
			if (colors[i] == colors[adjacency_list[i][j]]) {
				passed = false;
				cout << "Failed coloring between nodes : " << i << " -- " << adjacency_list[i][j] << endl;
				fflush(stdin);
				fflush(stdout);
				break;
			}
		}
	}
	return passed;
}

int main(int argc, char *argv[]) {
	int nvertices, max_degree;
	int *num_edges;
	int **adjacency_list;
	char *filename = argv[1];

	cout << filename << endl;

	ifstream fin(filename);
	fin >> (max_degree);
	fin >> (nvertices);
	cout << nvertices << " : " << max_degree << endl;
	fflush(stdin);
	fflush(stdout);
	adjacency_list = (int **) malloc(nvertices * sizeof(int *));
	num_edges = (int *) malloc(nvertices * sizeof(int ));
	for (int i = 0; i < nvertices; ++i) {
		fin >> num_edges[i];
		fflush(stdin);
		fflush(stdout);
		adjacency_list[i] = (int *) malloc(num_edges[i] * sizeof(int));
		for (int j = 0; j < num_edges[i]; ++j) {
			fin >> adjacency_list[i][j];
			fflush(stdin);
			fflush(stdout);
		}
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

	double begin = currentSeconds();
	int *colors = IPGC(nvertices, num_edges, max_degree, adjacency_list);
	double end = currentSeconds();
	double timeSec = (end - begin);

	cout << "Coloring done!" << endl;
	cout << "Time for coloring : " << timeSec * 1000 << " ms" << endl;

	if (checker(nvertices, max_degree, num_edges, colors, adjacency_list)) {
		cout << "CORRECT COLORING!!!" << endl;
	} else {
		cout << "INCORRECT COLORING!!!" << endl;
	}

	// Free all arrays
	free(num_edges);
	for (int i = 0; i < nvertices; ++i) {
		free(adjacency_list[i]);
	}
	free(adjacency_list);
}