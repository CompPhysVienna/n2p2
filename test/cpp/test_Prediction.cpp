#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Prediction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"
#include "ExamplePrediction.h"
#include "tolerance.h"

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

Tolerance const tolerance =
    Tolerance(1000.0 * numeric_limits<double>::epsilon());

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
    REQUIRE_CLOSE_TO_REFERENCE(example.energy, p.structure.energy, tolerance);

    for (size_t i = 0; i < p.structure.atoms.size(); ++i)
    {
        Vec3D const& f = p.structure.atoms.at(i).f;
        Vec3D const& fRef = example.forces.at(i);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fx");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[0], f[0], tolerance);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fy");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[1], f[1], tolerance);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fz");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[2], f[2], tolerance);
    }

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
