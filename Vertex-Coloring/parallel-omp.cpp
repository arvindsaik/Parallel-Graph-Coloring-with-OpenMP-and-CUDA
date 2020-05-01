#include <bits/stdc++.h>
#include <omp.h>


using namespace std;
#define NUM_THREADS 10

void printGraph(int n, int m[], int **adj_list) {
	for (int i = 0; i < n; ++i) {
		cout << "Node " << i << " || ";
		cout << m[i] << endl;
		fflush(stdin);
		fflush(stdout);
		for (int j = 0; j < m[i]; ++j) {
			cout << i << " : " << j << endl;
			fflush(stdin);
			fflush(stdout);
			cout << adj_list[i][j] << "    ";
			fflush(stdin);
			fflush(stdout);
		}
		cout << endl;
		fflush(stdin);
		fflush(stdout);
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

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int **adj_list, int *colors) {
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (int i = 0; i < num_conflicts; ++i) {
		bool *forbidden = (bool *) malloc(num_conflicts * sizeof(bool));
		for (int j = 0; j < maxd + 1; ++j) {
			forbidden[j] = false;
		}
		int v = conflicts[i];
		for (int j = 0; j < m[v]; ++j) {
			int u = adj_list[v][j];
			if (colors[u] >= 0)
				forbidden[colors[u]] = true;
		}
		for (int j = 0; j < maxd + 1; ++j) {
			if (forbidden[j] == false) {
				colors[v] = j;
				break;
			}
		}
		free(forbidden);
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
				temp_conflicts[*temp_num_conflicts] = u;
				*temp_num_conflicts = *temp_num_conflicts + 1;
				colors[u] = -u;
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
		cout << "Assign colors done!\n";
		detect_conflicts(num_conflicts, conflicts, m, adj_list, colors, &temp_num_conflicts, temp_conflicts);
		cout << "Detect conflicts done\n";
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

bool checker(int nvertices, int *num_edges, int *colors, int **adjacency_list) {
	bool passed = true;
	for (int i = 0; i < nvertices; ++i) {
		for (int j = 0; j < num_edges[i]; ++j) {
			if (colors[i] == colors[adjacency_list[i][j]]) {
				passed = false;
				cout << "Failed coloring between nodes : " << i << " -- " << adjacency_list[i][j];
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
		cout << endl;
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

	int *colors = IPGC(nvertices, num_edges, max_degree, adjacency_list);
	cout << "Coloring done!" << endl;
	if (checker(nvertices, num_edges, colors, adjacency_list)) {
		cout << "CORRECT COLORING!!!" << endl;
	} else {
		cout << "INCORRECT COLORING!!!" << endl;
	}
}