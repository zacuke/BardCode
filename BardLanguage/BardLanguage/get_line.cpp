// get_line.cpp
#include <iostream>
#include <string>
#include "get_line.h"

string get_line(istream& is) {
    string line;
    char c;
    while ((c = is.get()) != EOF && c != '\n') {
        line += c;
    }

    if (c == '\n') {
        line += c;
    }

    return line;
}
