// format_string.cpp
#include "format_string.h"

string format_string(const char* format, const string& value) {
	char buffer[100];
	std::snprintf(buffer, sizeof(buffer), format, value);
	return string(buffer);
}