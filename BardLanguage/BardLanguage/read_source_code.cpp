// read_source_code.cpp
#include <iostream>
#include <fstream>
#include "read_source_code.h"
#include "get_line.h"

string read_source_code(string filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string source_code;
    string line;
    while (getline(infile, line)) {
        source_code += line + "\n";
    }

    infile.close();
    return source_code;
}

