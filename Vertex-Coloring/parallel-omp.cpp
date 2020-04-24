#include <bits/stdc++.h>
#include <omp.h>


using namespace std;

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

void loadGraph(char *filename, int *nvertices, int *max_degree, int *num_edges, int **adjacency_list) {

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
	bool **forbidden = (bool **) malloc(num_conflicts * sizeof(bool *));
	for (int i = 0; i < num_conflicts; ++i) {
		forbidden[i] = (bool *) malloc((maxd + 1) * sizeof(bool));
	}
#if OMP
	#pragma omp parallel num_threads(4)
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
		cout << i << "-----";
		fin >> num_edges[i];
		cout << num_edges[i] << " : ";
		fflush(stdin);
		fflush(stdout);
		adjacency_list[i] = (int *) malloc(num_edges[i] * sizeof(int));
		for (int j = 0; j < num_edges[i]; ++j) {
			fin >> adjacency_list[i][j];
			cout << adjacency_list[i][j] << " ";
			fflush(stdin);
			fflush(stdout);
		}
		cout << endl;
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

//	printGraph(nvertices, num_edges, adjacency_list);
	IPGC(nvertices, num_edges, max_degree, adjacency_list);
}