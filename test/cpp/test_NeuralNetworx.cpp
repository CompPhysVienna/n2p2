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
#include <algorithm> // std::fill, std::generate
#include <cstddef>   // std::size_t
#include <iostream>
#include <numeric>   // std::iota
#include <random>    // std::mt19937_64
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();
double const delta = 1.0E-5;

BoostDataContainer<ExampleNeuralNetworx> containerNN;
BoostDataContainer<ExampleActivation> containerActivation;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_DATA_TEST_CASE(Initialize_CorrectNetworkArchitecture,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);

    BOOST_REQUIRE_EQUAL(nn.getNumLayers(), example.numLayers);
    BOOST_REQUIRE_EQUAL(nn.getNumConnections(), example.numConnections);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), example.numWeights);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), example.numBiases);
}

BOOST_DATA_TEST_CASE(InitializeWithStrings_CorrectNetworkArchitecture,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer,
                     example.activationStringPerLayer);

    BOOST_REQUIRE_EQUAL(nn.getNumLayers(), example.numLayers);
    BOOST_REQUIRE_EQUAL(nn.getNumConnections(), example.numConnections);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), example.numWeights);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), example.numBiases);
}

BOOST_DATA_TEST_CASE(SetAndGetConnections_OriginalValuesRestored,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co(nn.getNumConnections(), 0);
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnections(ci);
    nn.getConnections(co);
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
}

BOOST_DATA_TEST_CASE(SetAndGetConnectionsAO_OriginalValuesRestored,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);

    vector<double> ci(nn.getNumConnections(), 0);
    vector<double> co(nn.getNumConnections(), 0);
    iota(ci.begin(), ci.end(), 0); 
    nn.setConnectionsAO(ci);
    nn.getConnectionsAO(co);
    BOOST_CHECK_EQUAL_COLLECTIONS(ci.begin(), ci.end(), co.begin(), co.end());
}

BOOST_DATA_TEST_CASE(SetAndWriteConnections_OriginalValuesRestored,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);

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
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);

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

BOOST_DATA_TEST_CASE(SingleNeuronActivation_AnalyticNumericDerivativesMatch,
                     bdata::make(containerActivation.examples),
                     example)
{
    // Set up NN with only two neurons.
    NeuralNetworx nn(vector<size_t>{1, 1},
                     vector<string>{"l", example.activation});
    nn.setConnections(vector<double>{1.0, 0.0});

    for (size_t i = 0; i < example.x.size(); ++i)
    {
        nn.propagate(vector<double>{example.x.at(i)});
        vector<double> output;
        nn.getOutput(output);
        cout << output.at(0) << "\n";
        BOOST_TEST_INFO(example.name + " x(" << i
                        << ") = " << example.x.at(i));
        BOOST_REQUIRE_SMALL(output.at(0) - example.f.at(i), accuracy);

        nn.propagate(vector<double>{example.x.at(i) - delta});
        double const yLeft = nn.getNeuronProperties(1, 0).at("y");

        nn.propagate(vector<double>{example.x.at(i) + delta});
        double const yRight = nn.getNeuronProperties(1, 0).at("y");

        nn.propagate(vector<double>{example.x.at(i)});
        auto yCenterProperties = nn.getNeuronProperties(1, 0);
        double const y = yCenterProperties.at("y");
        BOOST_REQUIRE_SMALL(y - example.f.at(i), accuracy);

        double const dydx = yCenterProperties.at("dydx");
        double const dydxNum = (yRight - yLeft) / (2.0 * delta);
        BOOST_REQUIRE_SMALL(dydx - dydxNum, 10 * delta);

        double const d2ydx2 = yCenterProperties.at("d2ydx2");
        double const d2ydx2Num = (yRight - 2.0 * y + yLeft) / (delta * delta);

        //cout << strpr("yl        : %23.16E\n", yLeft);
        //cout << strpr("y         : %23.16E\n", y);
        //cout << strpr("f         : %23.16E\n", example.f.at(i));
        //cout << strpr("yr        : %23.16E\n", yRight);
        //cout << strpr("yr-2y+yl  : %23.16E\n", yRight - 2.0 * y + yLeft);
        //cout << strpr("delta**2  : %23.16E\n", delta * delta);
        //cout << "\n";
        //cout << strpr("dydx      : %23.16E\n", dydx);
        //cout << strpr("dydxNum   : %23.16E\n", dydxNum);
        //cout << "\n";
        //cout << strpr("d2ydx2    : %23.16E\n", d2ydx2);
        //cout << strpr("d2ydx2Num : %23.16E\n", d2ydx2Num);

        BOOST_REQUIRE_SMALL(d2ydx2 - d2ydx2Num, 10 * delta);
    }
}

BOOST_AUTO_TEST_CASE(ComputeInputDerivative_AnalyticNumericDerivativesMatch)
{
    size_t numInput = 3;
    size_t numOutput = 4;
    NeuralNetworx nn(vector<size_t>{numInput, 4, 5, numOutput},
                     vector<string>{"l", "p", "p", "p"});

    mt19937_64 rng(0);
    uniform_real_distribution<double> distribution(-1.0, 1.0);
    auto generator = [&distribution, &rng]() { return distribution(rng); };

    vector<double> connections(nn.getNumConnections());
    generate(connections.begin(), connections.end(), generator);
    nn.setConnections(connections);

    vector<double> input(numInput);
    generate(input.begin(), input.end(), generator);
    nn.setInput(input);

    nn.propagate(true);
    vector<vector<double>> derivInput;
    nn.getDerivInput(derivInput);

    vector<double> outputLeft;
    vector<double> outputRight;
    vector<vector<double>> derivInputNum(derivInput.size());
    for (size_t i = 0; i < numInput; ++i)
    {
        vector<double> tmpInput = input;
        tmpInput.at(i) -= delta;
        nn.setInput(tmpInput);
        nn.propagate();
        nn.getOutput(outputLeft);

        tmpInput.at(i) += 2.0 * delta;
        nn.setInput(tmpInput);
        nn.propagate();
        nn.getOutput(outputRight);
        for (size_t o = 0; o < numOutput; ++o)
        {
            derivInputNum.at(o).push_back(
                (outputRight.at(o) - outputLeft.at(o)) / (2.0 * delta));
        }
    }

    for (size_t o = 0; o < numOutput; ++o)
    {
        for (size_t i = 0; i < numInput; ++i)
        {
            BOOST_TEST_INFO("dy_out(" << o << ")/dy_in(" << i << ")");
            BOOST_REQUIRE_SMALL(derivInput.at(o).at(i)
                                - derivInputNum.at(o).at(i),
                                10 * delta);
        }
    }

    //for (auto l : nn.info()) cout << l;

    //for (auto on : derivInputNum)
    //{
    //    for (auto in : on)
    //    {
    //        cout << in << " ";
    //    }
    //    cout << "\n";
    //}

    //cout << "\n";

    //for (auto on : derivInput)
    //{
    //    for (auto in : on)
    //    {
    //        cout << in << " ";
    //    }
    //    cout << "\n";
    //}
}

BOOST_DATA_TEST_CASE(GetLayerLimits_CorrectLimits,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);
    auto limits = nn.getLayerLimits();
    if (example.limitsLayers.size() > 0)
    {
        BOOST_REQUIRE_EQUAL(limits.size(), example.limitsLayers.size());
    }
}

BOOST_DATA_TEST_CASE(GetNeuronLimits_CorrectLimits,
                     bdata::make(containerNN.examples),
                     example)
{
    NeuralNetworx nn(example.numNeuronsPerLayer, example.activationPerLayer);
    auto limits = nn.getNeuronLimits();
    if (example.limitsNeurons.size() > 0)
    {
        //BOOST_REQUIRE_EQUAL(limits.size(), example.limitsNeurons.size());
    }
}

BOOST_AUTO_TEST_SUITE_END()
