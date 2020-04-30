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

void assign_colors(int n, int maxd, int *colors, bool **vforbidden) {
	cout << "Max degree is : " << maxd <<  endl;
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
			if (smaller_vertex == 45 || bigger_vertex == 45) {
				cout << smaller_vertex  << " < " << bigger_vertex << endl;
				cout << colors[smaller_vertex] << " : " << colors[bigger_vertex] << endl;
			}
			temp_colors[smaller_vertex] = -1;
			is_conflict = true;
		}
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
	bool *conflicts = (bool *) malloc(n * sizeof(bool));
	long iter = 1;
	int is_conflict = true;
	while (is_conflict) {
		cout << "Iteration " << iter++ << endl;
		assign_colors(n, maxd, colors, vforbidden);
		for (int i = 0; i < n; ++i) {
			cout << "Color of node " << i << " : " << colors[i] << endl;
		}
		for (int i = 0; i < n; ++i) {
			memset(vforbidden[i], 0, sizeof(vforbidden[i]));
			temp_colors[i] = colors[i];
		}
		is_conflict = detect_conflicts(num_edges, edges, colors, temp_colors, vforbidden);
		for (int i = 0; i < n; ++i) {
			colors[i] = temp_colors[i];
			cout << "Color of node " << i << " : " << colors[i] << endl;
		}
//		if (iter == 10) break;
	}

	for (int i = 0; i < n; ++i) {
		cout << "Color of node " << i << " : " << colors[i] << endl;
	}
	fflush(stdin);
	fflush(stdout);
	return colors;
}

bool checker(int num_edges, int **edges, int *colors) {
	bool passed = true;
	for (int i = 0; i < num_edges; ++i) {
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
		cout << edges[i][0] << "---" << edges[i][1] << endl;
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

	int *colors = IPGC(nvertices, num_edges, max_degree, edges);

	cout << "Coloring done!" << endl;
	if (checker(num_edges, edges, colors)) {
		cout << "CORRECT COLORING!!!" << endl;
	} else {
		cout << "INCORRECT COLORING!!!" << endl;
	}
}