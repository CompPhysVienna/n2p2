#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-atomenv
#include "Example_nnp_atomenv.h"
#include "nnp_test.h"
#include "utility.h"

#include <limits> // std::numeric_limits
#include <map> // std::map

using namespace std;
using namespace nnp;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_atomenv> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_atomenv const example)
{
    Example_nnp_atomenv const& ex = example;

    BOOST_REQUIRE(bfs::exists("nnp-atomenv.log.0000"));
    BOOST_REQUIRE(bfs::exists("atomic-env.G"));
    BOOST_REQUIRE(bfs::exists("atomic-env.dGdx"));
    BOOST_REQUIRE(bfs::exists("atomic-env.dGdy"));
    BOOST_REQUIRE(bfs::exists("atomic-env.dGdz"));
    BOOST_REQUIRE(bfs::exists("neighbors.histo"));

    map<string, size_t> nexpected;
    map<string, size_t> nexpectedFull;
    for (auto e1 : ex.elements)
    {
        nexpected    [e1] = 1 + ex.numSF.at(e1);
        nexpectedFull[e1] = 1 + ex.numSF.at(e1);
        for (auto e2 : ex.elements)
        {
            nexpected.at(e1)     += ex.neighCut.at(make_pair(e1, e2)).first
                                  * ex.neighCut.at(make_pair(e2, e1)).second;
            nexpectedFull.at(e1) += ex.neighCut.at(make_pair(e1, e2)).first
                                  * ex.numSF.at(e2);
        }
    }

    string line;
    ifstream file;
    file.open("atomic-env.G");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));;
        string element = columns.at(0);
        BOOST_REQUIRE_EQUAL(columns.size(), nexpectedFull.at(element));
    }
    file.close();

    file.open("atomic-env.dGdx");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));;
        string element = columns.at(0);
#ifndef NNP_FULL_SFD_MEMORY
        BOOST_REQUIRE_EQUAL(columns.size(), nexpected.at(element));
#else
        BOOST_REQUIRE_EQUAL(columns.size(), nexpectedFull.at(element));
#endif
    }
    file.close();

    return;
}
