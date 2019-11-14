#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Prediction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Atom.h"
#include "Structure.h"
#include "Prediction.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;
using namespace boost::filesystem;

namespace bdata = boost::unit_test::data;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

vector<string> sampleDirectories = {"../../examples/nnp-predict/H2O_RPBE-D3",
                                    "../../examples/nnp-predict/Cu2S_PBE"};

vector<double> energies = {-2.7564547347815904E+04,
                           -5.7365603183874578E+02};


BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE(PredictStructure_CorrectEnergiesForces,
                     bdata::make(sampleDirectories) ^ bdata::make(energies),
                     sampleDirectory,
                     energy)
{
    BOOST_REQUIRE(copy_directory_recursively(sampleDirectory, "test"));
    current_path("test");

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    BOOST_REQUIRE_SMALL(p.structure.energy - energy, accuracy);

    current_path("..");
    remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
