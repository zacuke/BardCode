// node.h
#pragma once
#include <string>
#include <vector>
using namespace std;
struct Node {
	string type;
	vector<Node*> children  ;
	string value;
	string name;
	string label;
	Node* parent = nullptr;

	Node(string type_) {
	 
		type = type_;
	}

};
 