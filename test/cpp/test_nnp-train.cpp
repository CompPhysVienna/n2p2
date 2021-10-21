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

//double const accuracy = 10.0 * numeric_limits<double>::epsilon();
double const accuracy = 1E-9;

BoostDataContainer<Example_nnp_train> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_train const example)
{
    bool epochFound = false;
    string line;
    ifstream file;

    string filename{"learning-curve.out"};
    if (example.args != "")
        filename += ".stage-" + example.args;

    file.open(filename);
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) == to_string(example.lastEpoch))
        {
            epochFound = true;
            if (example.args == "1")
            {
                BOOST_REQUIRE_SMALL(example.rmseChargesTrain - stod(columns.at(1)),
                                    accuracy);
                BOOST_REQUIRE_SMALL(example.rmseChargesTest - stod(columns.at(2)),
                                    accuracy);
            }
            else
            {
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
    }
    BOOST_REQUIRE_MESSAGE(epochFound,
                          string("ERROR: Epoch information was not "
                                 "found in file."));

    return;
}
