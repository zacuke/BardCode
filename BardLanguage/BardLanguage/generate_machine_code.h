// generate_machine_code.h
#pragma once
#include <vector>
#include <string>
#include "node.h"

using namespace std;
 
vector<string> generate_machine_code(Node* node, string& target_machine_architecture);
 