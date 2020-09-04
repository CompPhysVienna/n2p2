#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <ostream> // std::ostream
#include <string>  // std::string

struct Example
{
    std::string name;
    std::string description;
};

std::ostream& operator<<(std::ostream& os, Example const& example)
{
    os << "Example name       : \"" << example.name << "\"\n";
    os << "Example description: \"" << example.description << "\"\n";

    return os;
}

#endif
