#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    string filename;
    filename = argv[1];
    ifstream fin(filename.c_str());

    while (fin.peek() == '%') fin.ignore(2048, '\n');

    long M, N, L;
    long vertices = 0, edges = 0;

    fin >> M >> N >> L;
    vector<vector<long> > adjacency_list(M, vector<long>(0));
    long max_degree = 0;

    for (int i = 0; i < L; i++) {
        long a, b;
        double d;
        fin >> a >> b >> d;
        a -= 1;
        b -= 1;
        if (a == b) continue;
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

    fflush(stdin);
    fflush(stdout);
    string new_filename = filename.substr(0, filename.length()-3);
    new_filename += "_graph";
    ofstream fout(new_filename.c_str());

    fout << max_degree << "\n";
    fout << edges << " " << vertices << "\n";
    for (long i = 0; i < M; i++) {
        fout << adjacency_list[i].size();
        for (long j = 0; j < adjacency_list[i].size(); j++) {
            fout << " ";
            fout << adjacency_list[i][j];
        }
        fout << "\n";
    }

    cout << "Format of the graph generated: " <<endl << "MAX DEGREE "<<endl << "NUM_EDGES<space>NUM_VERTICES" <<endl ;
    cout << "Adjacency List Size for vertex 0<space>Adjacency list for vertex 0" <<endl<<"....."<<endl <<endl;
    cout << "Please note: The graph generated is an undirected graph" << endl;
    return 0;
}