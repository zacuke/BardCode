// append_vector_string.cpp
#include "append_vector_string.h"
#include <vector>
#include <string>
 
void append_vector_string(vector<string>& machine_code, const vector<string>& other_machine_code) {
    for (const string& instruction : other_machine_code) {
        machine_code.push_back(instruction);
    }
}