#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNetworx
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"
#include "ExampleNeuralNetworx.h"

#include "NeuralNetworx.h"
#include "utility.h"
#include <algorithm> // std::fill
#include <cstddef>   // std::size_t
#include <iostream>
#include <numeric>   // std::iota
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

//double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BoostDataContainer<ExampleNeuralNetworx> container;

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE(Initialize_CorrectNetworkArchitecture,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationPerLayer);

    BOOST_REQUIRE_EQUAL(nn.getNumConnections(), example.numConnections);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), example.numWeights);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), example.numBiases);
}

BOOST_DATA_TEST_CASE(InitializeWithStrings_CorrectNetworkArchitecture,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationStringPerLayer);

    BOOST_REQUIRE_EQUAL(nn.getNumConnections(), example.numConnections);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), example.numWeights);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), example.numBiases);
}

BOOST_DATA_TEST_CASE(SetAndGetConnections_OriginalValuesRestored,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co(nn.getNumConnections(), 0);
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnections(ci);
    nn.getConnections(co);
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
}

BOOST_DATA_TEST_CASE(SetAndGetConnectionsAO_OriginalValuesRestored,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co(nn.getNumConnections(), 0);
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnectionsAO(ci);
    nn.getConnectionsAO(co);
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
}

BOOST_DATA_TEST_CASE(SetAndWriteConnections_OriginalValuesRestored,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co;
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnections(ci);
    ofstream ofile;
    ofile.open("weights.data");
    nn.writeConnections(ofile);
    ofile.close();
    ifstream ifile;
    ifile.open("weights.data");
    BOOST_REQUIRE(ifile.is_open());
    string line;
    while (getline(ifile, line))
    {
        if (line.at(0) != '#')
        {
            vector<string> splitLine = split(reduce(line));
            co.push_back(atof(splitLine.at(0).c_str()));
        }
    }
    
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
    bfs::remove_all("weights.data");
}

BOOST_DATA_TEST_CASE(SetAndWriteConnectionsAO_OriginalValuesRestored,
                     bdata::make(container.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co;
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnectionsAO(ci);
    ofstream ofile;
    ofile.open("weights.data");
    nn.writeConnectionsAO(ofile);
    ofile.close();
    ifstream ifile;
    ifile.open("weights.data");
    BOOST_REQUIRE(ifile.is_open());
    string line;
    while (getline(ifile, line))
    {
        if (line.at(0) != '#')
        {
            vector<string> splitLine = split(reduce(line));
            co.push_back(atof(splitLine.at(0).c_str()));
        }
    }
    
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
    bfs::remove_all("weights.data");
}

BOOST_AUTO_TEST_SUITE_END()
