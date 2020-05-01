#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    string filename;
    filename = argv[1];
    int flag = atoi(argv[2]);
    ifstream fin(filename.c_str());

    while (fin.peek() == '%') fin.ignore(2048, '\n');

    long M, N, L;
    long vertices = 0, edges = 0;
    set<pair<long, long> > edges_set;

    fin >> M >> N >> L;
    vector<vector<long> > adjacency_list(M, vector<long>(0));
    long max_degree = 0;
	cout << "Num vertices: " << M << endl;
    for (int i = 0; i < L; i++) {
        long a, b;
        double d;
        fin >> a >> b;
        if (flag == 0) fin >> d;
        a -= 1;
        b -= 1;
        if (a == b) continue;
        if (edges_set.find(make_pair(a, b)) != edges_set.end() || edges_set.find(make_pair(b, a)) != edges_set.end()) {
            continue;
        } else {
            edges_set.insert(make_pair(a, b));
        }
        adjacency_list[a].push_back(b);
        adjacency_list[b].push_back(a);
        if (adjacency_list[a].size() > max_degree) {
            max_degree = adjacency_list[a].size();
        }
        if (adjacency_list[b].size() > max_degree) {
            max_degree = adjacency_list[b].size();
        }
        edges++;
    }

    vertices = M;

    string new_filename = filename.substr(0, filename.length()-4);
    new_filename += "_graph";
    ofstream fout(new_filename.c_str());

    fflush(stdin);
    fflush(stdout);
    fout << max_degree << "\n";
    fout << vertices << "\n";
    for (long i = 0; i < M; i++) {
        fout << adjacency_list[i].size();
        for (long j = 0; j < adjacency_list[i].size(); j++) {
            fout << " ";
            fout << adjacency_list[i][j];
        }
        fout << "\n";
    }
	fout.close();
    cout << "Format of the graph generated: " <<endl << "MAX DEGREE "<<endl << "NUM_VERTICES" <<endl ;
    cout << "Adjacency List Size for vertex 0<space>Adjacency list for vertex 0" <<endl<<"....."<<endl <<endl;
    cout << "Please note: The graph generated is an undirected graph" << endl;
    return 0;
}