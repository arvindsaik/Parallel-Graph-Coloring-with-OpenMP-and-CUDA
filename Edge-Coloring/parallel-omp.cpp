#include <bits/stdc++.h>
#include <omp.h>
#include <ctime>


using namespace std;
#define NUM_THREADS 10

void assign_colors(long n, long maxd, long *colors, bool **vforbidden) {
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (long i = 0; i < n; ++i) {
		if (colors[i] != -1) continue;
		for (long j = 0; j < maxd + 1; ++j) {
			if (vforbidden[i][j] == false) {
				colors[i] = j;
				break;
			}
		}
	}
}

bool detect_conflicts(long num_edges, long **edges, long *colors, long *temp_colors, bool **vforbidden) {
	bool is_conflict = false;
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
	for (long  i = 0; i < num_edges; ++i) {
		long smaller_vertex, bigger_vertex;
		if (edges[i][0] > edges[i][1]) {
			bigger_vertex = edges[i][0];
			smaller_vertex = edges[i][1];
		} else {
			bigger_vertex = edges[i][1];
			smaller_vertex = edges[i][0];
		}
		if (colors[smaller_vertex] == colors[bigger_vertex]) {
			temp_colors[smaller_vertex] = -1;
			is_conflict = true;
		}
		vforbidden[smaller_vertex][colors[bigger_vertex]] = true;
	}
	return is_conflict;
}

long *IPGC(long n, long num_edges, long maxd, long **edges) {
	long *colors = (long *) malloc(n * sizeof(long));
	long *temp_colors = (long *) malloc(n * sizeof(long));
	for (long i = 0; i < n; ++i) {
		colors[i] = -1;
		temp_colors[i] = -1;
	}
	bool **vforbidden = (bool **) malloc(n * sizeof(bool *));
	for (long i = 0; i < n; ++i) {
		vforbidden[i] = (bool *) calloc(maxd + 1, sizeof(bool));
	}
	bool *conflicts = (bool *) malloc(n * sizeof(bool));
	long iter = 0;
	long is_conflict = true;
	while (is_conflict) {
		iter++;
		fflush(stdin);
		fflush(stdout);
		assign_colors(n, maxd, colors, vforbidden);

		fflush(stdin);
		fflush(stdout);
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
		for (long i = 0; i < n; ++i) {
			memset(vforbidden[i], 0, sizeof(vforbidden[i]));
			temp_colors[i] = colors[i];
		}


		is_conflict = detect_conflicts(num_edges, edges, colors, temp_colors, vforbidden);

		fflush(stdin);
		fflush(stdout);
#if OMP
	#pragma omp parallel num_threads(NUM_THREADS)
	#pragma omp for
#endif
		for (long i = 0; i < n; ++i) {
			colors[i] = temp_colors[i];
		}
	}
	cout << "Iterations taken to converge " << iter << endl;
	for (long i = 0; i < n; ++i) {
		cout << "Color of node " << i << " : " << colors[i] << endl;
	}
	fflush(stdin);
	fflush(stdout);
	return colors;
}

bool checker(long num_edges, long **edges, long *colors) {
	bool passed = true;
	for (long i = 0; i < num_edges; ++i) {
		if (colors[edges[i][0]] == colors[edges[i][1]]) {
			passed = false;
		}
	}
	return passed;
}

int main(int argc, char *argv[]) {
	long nvertices;
	long num_edges;
	long max_degree;
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
	long **edges = (long **) malloc(num_edges * sizeof(long *));
	for (long i = 0; i < num_edges; ++i) {
		edges[i] = (long *) malloc(2 * sizeof(long));
	}

	for (long i = 0; i < num_edges; ++i) {
		fin >> edges[i][0];
		fin >> edges[i][1];
		fflush(stdin);
		fflush(stdout);
	}
	fin.close();

	clock_t begin = clock();
	// Perform coloring
	long *colors = IPGC(nvertices, num_edges, max_degree, edges);
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );

	cout << "Time for coloring : " << timeSec << endl;

	// Call checker
	if (checker(num_edges, edges, colors)) {
		cout << "CORRECT COLORING!!!" << endl;
	} else {
		cout << "INCORRECT COLORING!!!" << endl;
	}
}