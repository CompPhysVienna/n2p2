#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE lammps-nnp
#include "Example_lammps_nnp.h"
#include "nnp_test.h"
#include "utility.h"

#include <fstream> // std::ifstream
#include <limits>  // std::numeric_limits
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_lammps_nnp> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_lammps_nnp const example)
{
    bool timeStepFound = false;
    bool startReading = false;
    string line;
    ifstream file;
    file.open("log.lammps");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.size() > 0 && columns.at(0) == "Step")
        {
            startReading = true;
        }
        if (startReading == true &&
            columns.size() >= 5 &&
            columns.at(0) == to_string(example.lastTimeStep))
        {
            timeStepFound = true;
            BOOST_REQUIRE_SMALL(example.potentialEnergy - stod(columns.at(2)),
                                accuracy);
            BOOST_REQUIRE_SMALL(example.totalEnergy - stod(columns.at(4)),
                                accuracy);
            break;
        }
    }
    BOOST_REQUIRE_MESSAGE(timeStepFound,
                          string("ERROR: Epoch information was not "
                                 "found in file."));

    return;
}
