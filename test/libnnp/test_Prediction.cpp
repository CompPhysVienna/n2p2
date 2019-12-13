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
#include "prediction_results.h"
#include "utility.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string, std::string::find_last_of
#include <vector> // std::vector

using namespace std;
using namespace nnp;
using namespace boost::filesystem;

namespace bdata = boost::unit_test::data;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

ReferenceData referenceData;

struct FixtureRepairDir
{
    FixtureRepairDir()
    {
        string p = current_path().string();
        if (p.substr(p.size() - 4) == "test")
        {
            current_path("..");
            remove_all("test");
        }
        else if (exists("test"))
        {
            remove_all("test");
        }
    }

};

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       PredictStructure_CorrectEnergiesForces,
                       bdata::make(referenceData.structureReferences),
                       ref)
{
    BOOST_REQUIRE(copy_directory_recursively(ref.path, "test"));
    current_path("test");

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    BOOST_TEST_INFO(ref.example + " Potential energy");
    BOOST_REQUIRE_SMALL(p.structure.energy - ref.energy, accuracy);

    for (size_t i = 0; i < p.structure.atoms.size(); ++i)
    {
        Vec3D d = p.structure.atoms.at(i).f - ref.forces.at(i);
        BOOST_TEST_INFO(ref.example + " Atom " << i << " fx");
        BOOST_REQUIRE_SMALL(d[0], accuracy);
        BOOST_TEST_INFO(ref.example + " Atom " << i << " fy");
        BOOST_REQUIRE_SMALL(d[1], accuracy);
        BOOST_TEST_INFO(ref.example + " Atom " << i << " fz");
        BOOST_REQUIRE_SMALL(d[2], accuracy);
    }

    current_path("..");
    remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
