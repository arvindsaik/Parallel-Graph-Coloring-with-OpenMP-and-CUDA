#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    string filename;
    filename = argv[1];
    int flag = atoi(argv[2]);
    ifstream fin(filename.c_str());

    while (fin.peek() == '%') fin.ignore(2048, '\n');

    long M, N, L;
    long edges = 0;
    set<pair<long, long> > edges_set;

    fin >> M >> N >> L;
    vector<vector<long> > adjacency_list(M, vector<long>(0));
    long max_degree = 0;
    cout << "Num vertices: " << M << endl;

    string new_filename = filename.substr(0, filename.length()-4);
    new_filename += "_edge_list";
    ofstream fout(new_filename.c_str());

    set<long> vertices;
    map<long, long> edge_count;

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
        vertices.insert(a);
        vertices.insert(b);
    }

	set < pair < long, long > > ::iterator it;

    for (int i = 0; i < vertices.size(); ++i) {
    	edge_count.insert(make_pair(i, 0));
    }

	for (it = edges_set.begin(); it != edges_set.end(); ++it) {
		edge_count[it->first] = edge_count[it->first] + 1;
		edge_count[it->second] = edge_count[it->second] + 1;
		cout << edge_count[it->first] << " snsdfjnejn\n";
	}

	long max_edges = 0;
	for (int i = 0; i < vertices.size(); ++i) {
		map<long, long>::iterator it = edge_count.find(i);
		if (it->second > max_edges) {
			max_edges = it->second;
		}
	}

	fout << max_edges << " ";
	fout << vertices.size() << " ";
    fout << edges_set.size() << endl;


    for (it = edges_set.begin(); it != edges_set.end(); ++it) {
	    fout << it->first << " " << it->second << "\n";
    }

    fout.close();
    cout << "Format of the graph generated: " <<endl ;
    cout << "MaxDegree NumVertices NumEdges" << endl;
    cout << "VERTEX1 VERTEX2" << endl;
    cout << "VERTEX3 VERTEX4" << endl;
    cout << "VERTEX5 VERTEX6" << endl;
    cout << "... <all edges>" << endl;
    return 0;
}