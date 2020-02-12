#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Prediction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"
#include "ExamplePrediction.h"

#include "Atom.h"
#include "Structure.h"
#include "Prediction.h"
#include "utility.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string, std::string::find_last_of
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BoostDataContainer<ExamplePrediction> container;

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       PredictStructure_CorrectEnergiesForces,
                       bdata::make(container.examples),
                       example)
{
    BOOST_REQUIRE(copy_directory_recursively(example.pathData, "test"));
    bfs::current_path("test");

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    BOOST_TEST_INFO(example.name + " Potential energy");
    BOOST_REQUIRE_SMALL(p.structure.energy - example.energy, accuracy);

    for (size_t i = 0; i < p.structure.atoms.size(); ++i)
    {
        Vec3D d = p.structure.atoms.at(i).f - example.forces.at(i);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fx");
        BOOST_REQUIRE_SMALL(d[0], accuracy);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fy");
        BOOST_REQUIRE_SMALL(d[1], accuracy);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fz");
        BOOST_REQUIRE_SMALL(d[2], accuracy);
    }

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
