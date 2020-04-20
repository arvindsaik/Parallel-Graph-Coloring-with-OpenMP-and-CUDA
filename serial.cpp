#include <bits/stdc++.h>
using namespace std;

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

void assign_colors(int num_conflicts, int *conflicts, int maxd, int *m, int **adj_list, int *colors) {
	bool *forbidden = (bool *) malloc((maxd + 1) * sizeof(bool));
	for (int i = 0; i < num_conflicts; ++i) {
		for (int j = 0; j < maxd; ++j) {
			forbidden[j] = false;
		}
		int v = conflicts[i];
		for (int j = 0; j < m[i]; ++j) {
			int u = adj_list[i][j];
			forbidden[colors[u]] = true;
		}
		for(int j = 0; j < maxd + 1; ++j) {
			if (forbidden[j] == false) {
				colors[i] = j;
				break;
			}
		}
	}
}

void detect_conflicts(int num_conflicts, int *conflicts, int *m, int **adj_list, int *colors,
		int temp_num_conflicts, int *temp_conflicts) {
	for (int i = 0; i < num_conflicts; ++i) {
		int v = conflicts[i];
		for (int j = 0; j < m[i]; ++j) {
			int u = adj_list[i][j];
			if (colors[u] == colors[v] && u < v) {
				temp_conflicts[temp_num_conflicts++] = u;
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

		detect_conflicts(num_conflicts, conflicts, m, adj_list, colors, temp_num_conflicts, temp_conflicts);

		// Swap
		num_conflicts = temp_num_conflicts;
		int *temp;
		temp = temp_conflicts;
		temp_conflicts = conflicts;
		conflicts = temp;
	}

	for (int i = 0; i < n; ++i) {
		cout << "Color of node " << i << " : " << colors[i] << endl;
	}
 }

int main() {
	int nvertices = 4;
	int num_edges[] = {2, 3, 2, 3};
	int max_degree = 3;
	int **adjacency_list = (int **) malloc(nvertices*sizeof(int *));
	for (int i = 0 ; i < nvertices; ++i) {
		adjacency_list[i] = (int *) malloc(num_edges[i] * sizeof(int));
	}

	setGraph(nvertices, num_edges, adjacency_list);
	printGraph(nvertices, num_edges, adjacency_list);
	IPGC(nvertices, num_edges, max_degree, adjacency_list);
}