// BardLanguage.cpp
#include <iostream>

#include "read_source_code.h"
#include "get_line.h"
#include "tokenize_bard_language.h"
#include "node.h"
#include "parse_tokens.h"
#include "generate_machine_code.h"

void print_tree_tokens(Node* root) {
	if (root == nullptr) {
		return;
	}

	for (Node* child : root->children) {
		print_tree_tokens(child);
	}

	cout << root->type << " ";
}

int main()
{

	string filename = "source.txt";
	string source_code = read_source_code(filename);
	//Node* root = new Node("root");
	string arch = "x86";


	vector<string> tokens = tokenize_bard_language(source_code);

	for (int i = 0; i < tokens.size(); i++) {
		string xtra = (tokens[i] == string("\n")) ? "" : " ";
		cout << tokens[i] << xtra;
	}

	cout << endl;

 	Node* parsed_result = parse_tokens(tokens);
	print_tree_tokens(parsed_result);
		
	vector<string> machine_code = generate_machine_code(parsed_result, arch);
	return 0;


}
