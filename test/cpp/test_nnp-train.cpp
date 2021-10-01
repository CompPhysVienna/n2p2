#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-train
#include "Example_nnp_train.h"
#include "nnp_test.h"
#include "utility.h"

#include <fstream> // std::ifstream
#include <limits>  // std::numeric_limits
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_train> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_train const example)
{
    bool epochFound = false;
    string line;
    ifstream file;

    ifstream source("nnp-train.log.0000", ios::binary);
    ofstream dest("../nnp-train.log.0000." + example.name, ios::binary);
    dest << source.rdbuf();
    source.close();
    dest.close();
    ifstream source2("learning-curve.out", ios::binary);
    ofstream dest2("../learning-curve.out." + example.name, ios::binary);
    dest2 << source2.rdbuf();
    source2.close();
    dest2.close();

    file.open("learning-curve.out");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) == to_string(example.lastEpoch))
        {
            epochFound = true;
            BOOST_REQUIRE_SMALL(example.rmseEnergyTrain - stod(columns.at(1)),
                                accuracy);
            BOOST_REQUIRE_SMALL(example.rmseEnergyTest - stod(columns.at(2)),
                                accuracy);
            BOOST_REQUIRE_SMALL(example.rmseForcesTrain - stod(columns.at(9)),
                                accuracy);
            BOOST_REQUIRE_SMALL(example.rmseForcesTest - stod(columns.at(10)),
                                accuracy);
        }
    }
    BOOST_REQUIRE_MESSAGE(epochFound,
                          string("ERROR: Epoch information was not "
                                 "found in file."));

    return;
}
