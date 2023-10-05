// generate_machine_code.cpp
#include "generate_machine_code.h"
#include "format_string.h"
#include "append_vector_string.h"
#include "node.h"

vector<string> generate_machine_code(Node* node, string& target_machine_architecture) {
    vector<string> machine_code;

    // Check the type of the node
    if (node->type == "define") {
        // Generate machine code to define the variable
   
        machine_code.push_back(format_string("mov eax, {}", node->value));
        machine_code.push_back(format_string("mov [{}], eax", node->name));
    }
    else if (node->type == "if") {
        // Generate machine code to evaluate the condition expression
        Node* x = node->children[0];
        append_vector_string(machine_code, generate_machine_code(x, target_machine_architecture));
       
        // Generate machine code to jump to the then-branch or else-branch
        if (node->children[1]->type == "block") {
            machine_code.push_back(format_string("je {}", node->children[1]->label));
        }
        else {
            machine_code.push_back(format_string("jmp {}", node->children[1]->label));
        }

        // Generate machine code for the then-branch
        append_vector_string(machine_code, generate_machine_code(node->children[2], target_machine_architecture));

        // Generate machine code for the else-branch
        append_vector_string(machine_code, generate_machine_code(node->children[3], target_machine_architecture));
    }
    else if (node->type == "while") {
        // Generate machine code to evaluate the condition expression
        append_vector_string(machine_code, generate_machine_code(node->children[0], target_machine_architecture));

        // Generate machine code to jump to the body of the loop
        machine_code.push_back(format_string("je {}", node->children[1]->label));

        // Generate machine code for the body of the loop
        append_vector_string(machine_code, generate_machine_code(node->children[2], target_machine_architecture));

        // Generate machine code to jump back to the condition expression
        machine_code.push_back(format_string("jmp {}", node->label));
    }
    else if (node->type == "call") {
        // Generate machine code to call the function
        machine_code.push_back(format_string("call {}", node->name));
    }
    else if (node->type == "return") {
        // Generate machine code to return from the function
        machine_code.push_back("ret");
    }
    else {
        // Generate machine code for the node's children
        for (Node* child : node->children) {
            append_vector_string(machine_code, generate_machine_code(child, target_machine_architecture));
        }
    }

    return machine_code;
}

