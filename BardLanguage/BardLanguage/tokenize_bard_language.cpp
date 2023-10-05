// tokenize.cpp
#include <vector>
#include <string>
#include "tokenize_bard_language.h"
bool is_start_of_line( int i, const string& source_code  ) {
    while (i > 0 && source_code[i - 1] == ' ') {
        i--;
    }
    return source_code[i - 1] == '\n';
}

vector<string> tokenize_bard_language(string source_code) {
    vector<string> tokens;
    string token;
    bool in_string = false;
    int indent_level = 0 ;
    int prev_indent_level = 0;
    int prev_prev_indent_level = 0;

    for (int i = 0; i < source_code.length(); i++) {
        char c = source_code[i];

        if (c == '"' && !in_string) {
            in_string = true;
        }
        else if (c == '"' && in_string) {
            in_string = false;
            tokens.push_back(token);
            token.clear();
        }
        else if (in_string) {
            token += c;
        }
        else if (c == ' ' || c == '\n' || c == '\t' || c == '(' || c == ')') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            if (c == '\n') {
                if (prev_indent_level < prev_prev_indent_level)  
                    for (int i = 0; i < prev_prev_indent_level - prev_indent_level; i++)  
                        tokens.push_back("DEDENT");
 
                prev_prev_indent_level = prev_indent_level;
                prev_indent_level = indent_level;
             
            }
            if (c != ' ') {
                tokens.push_back(string(1, c));
            }
            else {
                if (!(indent_level % 2 == 0) && is_start_of_line(i, source_code)) {
                    tokens.push_back("INDENT");
                    prev_indent_level++; 
                    indent_level++;
                }
                else if (is_start_of_line(i, source_code)) {
                     indent_level--;
                }
            }          
        }
        else {
            token += c;
        }
    }

    if (!token.empty()) {
        tokens.push_back(token);
    }

    return tokens;
}
