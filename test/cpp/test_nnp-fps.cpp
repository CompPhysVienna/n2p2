#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-fps
#include "Example_nnp_fps.h"
#include "nnp_test.h"
#include "utility.h"

#include <fstream> // std::ifstream
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

BoostDataContainer<Example_nnp_fps> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_fps const example)
{
    int i = 0;
    string line;
    ifstream file;
    file.open("nnp-fps.log.0000");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.size() > 0 && columns.at(0) == "chosen")
        {
            BOOST_REQUIRE_EQUAL(example.chosenStructures.at(i),
                                stoi(columns.at(2)));
            i++;
        }
    }
    BOOST_REQUIRE_MESSAGE((size_t)i == example.chosenStructures.size(),
                          string("ERROR: Not all structures were selected."));

    return;
}
