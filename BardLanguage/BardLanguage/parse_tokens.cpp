// parse_tokens.cpp
#include <vector>
#include <string>
#include "node.h"
#include <stack>
 
Node* parse_tokens(vector<string> tokens) {
    // Create a stack to keep track of the current node and its ancestors.
    std::stack<Node*> node_stack;

    // Initialize the root node.
    Node* root = new Node("root");
    node_stack.push(root);

    // Iterate over the tokens and parse them into the tree structure.
    for (const string& token : tokens) {
        // If the token is a left parenthesis, create a new node and push it onto the stack.
        if (token == "(") {
            Node* new_node = new Node("node");
            new_node->parent = node_stack.top();
            node_stack.top()->children.push_back(new_node);
            node_stack.push(new_node);
        }

        // If the token is a right parenthesis, pop the current node from the stack.
        else if (token == ")") {
            node_stack.pop();
        }

        // Otherwise, the token is a value, a keyword, or a name.
        else {
            // If the token is a keyword, create a new node with the corresponding type.
            if (token == "if" || token == "define" || token == "else" || token == "lambda" || token == "print") {
                Node* new_node = new Node(token);
                new_node->parent = node_stack.top();
                node_stack.top()->children.push_back(new_node);
                node_stack.push(new_node);
            }

            // Otherwise, the token is a value or a name.
            else {
                Node* new_node = new Node("value");
                new_node->value = token;
                new_node->parent = node_stack.top();
                node_stack.top()->children.push_back(new_node);
            }
        }
    }

    // Return the root node of the tree.
    return root;
}