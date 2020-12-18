#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-predict
#include "Example_nnp_predict.h"
#include "nnp_test.h"
#include "utility.h"

#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

//double const accuracy = 10.0 * numeric_limits<double>::epsilon();
double const accuracy = 1E-5; // Output

BoostDataContainer<Example_nnp_predict> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_predict const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-predict.log"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }
    ifstream file;
    file.open("output.data");
    string line;
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) != "energy") continue;
        else
        {
            BOOST_REQUIRE_SMALL(example.energy - stod(columns.at(1)),
                                accuracy);
            break;
        }
    }
    file.close();

    return;
}
