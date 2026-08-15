#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNetwork
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "NeuralNetwork.h"
#include <cmath>     // std::sqrt, std::pow, std::tanh
#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream, std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-6;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Architecture_CorrectCounts)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};

    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    BOOST_REQUIRE_EQUAL(nn.getNumNeurons(), 2 + 3 + 1);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), 2 * 3 + 3 * 1);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), 3 + 1);
    BOOST_REQUIRE_EQUAL(nn.getNumConnections(),
                        nn.getNumWeights() + nn.getNumBiases());
}

BOOST_AUTO_TEST_CASE(SetGetConnections_RoundTrip)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> in(n);
    for (int i = 0; i < n; ++i) in.at(i) = 0.1 * (i + 1);
    nn.setConnections(in.data());

    vector<double> out(n, 0.0);
    nn.getConnections(out.data());
    for (int i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_SMALL(in.at(i) - out.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(Propagate_LinearNetwork_CorrectOutput)
{
    // All-identity 2-2-1 network: output is a plain linear map, computed
    // here independently and compared against propagate()'s result.
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 2, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    // Connection order (see NeuralNetwork::setConnections()): for each
    // layer, weights ordered by [source neuron][dest neuron], then biases
    // ordered by [dest neuron].
    vector<double> connections = {
        1.0, 2.0,   // x0 -> h0, x0 -> h1
        3.0, 4.0,   // x1 -> h0, x1 -> h1
        0.5, -0.5,  // bias h0, bias h1
        2.0, -1.0,  // h0 -> E, h1 -> E
        0.25        // bias E
    };
    nn.setConnections(connections.data());

    double const x0 = 1.5;
    double const x1 = -2.0;
    double input[2] = {x0, x1};
    nn.setInput(input);
    nn.propagate();

    double const h0 = 1.0 * x0 + 3.0 * x1 + 0.5;
    double const h1 = 2.0 * x0 + 4.0 * x1 + (-0.5);
    double const E  = 2.0 * h0 + (-1.0) * h1 + 0.25;

    double output = 0.0;
    nn.getOutput(&output);
    BOOST_REQUIRE_SMALL(output - E, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateDEdG_MatchesNumericDerivative_OneHiddenLayer)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(42);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    vector<double> dEdG(2);
    nn.calculateDEdG(dEdG.data());

    double const h = 1.0E-6;
    for (int i = 0; i < 2; ++i)
    {
        double const orig = input[i];

        input[i] = orig + h;
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        input[i] = orig - h;
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        input[i] = orig;

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdG.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDEdG_MatchesNumericDerivative_TwoHiddenLayers)
{
    int const numLayers = 4;
    int const numNeuronsPerLayer[4] = {3, 4, 3, 1};
    NeuralNetwork::ActivationFunction const af[4] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(7);

    double input[3] = {0.2, -0.4, 0.6};
    nn.setInput(input);
    nn.propagate();

    vector<double> dEdG(3);
    nn.calculateDEdG(dEdG.data());

    double const h = 1.0E-6;
    for (int i = 0; i < 3; ++i)
    {
        double const orig = input[i];

        input[i] = orig + h;
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        input[i] = orig - h;
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        input[i] = orig;

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdG.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDEdc_MatchesNumericDerivative)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(123);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    int const numConnections = nn.getNumConnections();
    vector<double> dEdc(numConnections);
    nn.calculateDEdc(dEdc.data());

    vector<double> conn(numConnections);
    nn.getConnections(conn.data());

    double const h = 1.0E-6;
    for (int i = 0; i < numConnections; ++i)
    {
        vector<double> connHi = conn;
        connHi.at(i) += h;
        nn.setConnections(connHi.data());
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        vector<double> connLo = conn;
        connLo.at(i) -= h;
        nn.setConnections(connLo.data());
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        nn.setConnections(conn.data());

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdc.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDFdc_MatchesNumericDerivative)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(99);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    vector<double> const dGdxyz = {1.0, 0.5};
    int const numConnections = nn.getNumConnections();
    vector<double> dFdc(numConnections);
    nn.calculateDFdc(dFdc.data(), dGdxyz.data());

    vector<double> conn(numConnections);
    nn.getConnections(conn.data());

    // Q(c) = dE/dG(c) . dGdxyz ; calculateDFdc() returns -dQ/dc (see "Note:
    // F = -dE/dx" in NeuralNetwork.cpp).
    auto computeQ = [&](vector<double> const& c) -> double
    {
        nn.setConnections(c.data());
        nn.setInput(input);
        nn.propagate();
        vector<double> dEdG(2);
        nn.calculateDEdG(dEdG.data());
        double Q = 0.0;
        for (size_t i = 0; i < dGdxyz.size(); ++i) Q += dEdG.at(i) * dGdxyz.at(i);
        return Q;
    };

    double const h = 1.0E-6;
    for (int i = 0; i < numConnections; ++i)
    {
        vector<double> cHi = conn;
        cHi.at(i) += h;
        vector<double> cLo = conn;
        cLo.at(i) -= h;
        double const numDeriv = (computeQ(cHi) - computeQ(cLo)) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dFdc.at(i) - (-numDeriv), accuracyNumeric);
    }

    nn.setConnections(conn.data());
}

BOOST_AUTO_TEST_CASE(ModifyConnections_ZeroBias)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_ZEROBIAS);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Layer 2->3 weights: 2*3=6 (idx 0-5), biases: 3 (idx 6-8).
    // Layer 3->4(out) weights: 3*1=3 (idx 9-11), bias: 1 (idx 12).
    int const biasIdx[] = {6, 7, 8, 12};
    for (int idx : biasIdx)
    {
        BOOST_REQUIRE_SMALL(out.at(idx), accuracy);
    }
    for (int i = 0; i < n; ++i)
    {
        bool const isBias = (i == 6 || i == 7 || i == 8 || i == 12);
        if (!isBias)
        {
            BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
        }
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_ZeroOutputWeights)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_ZEROOUTPUTWEIGHTS);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Output layer weights are idx 9-11 (see layout above).
    int const outWeightIdx[] = {9, 10, 11};
    for (int idx : outWeightIdx)
    {
        BOOST_REQUIRE_SMALL(out.at(idx), accuracy);
    }
    for (int i = 0; i < n; ++i)
    {
        bool const isOutWeight = (i == 9 || i == 10 || i == 11);
        if (!isOutWeight)
        {
            BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
        }
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_FanIn_OnlyScalesTanhLayers)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_FANIN);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Hidden layer (TANH) weights idx 0-5: divided by sqrt(numNeuronsPrevLayer=2).
    double const expectedHidden = 1.0 / sqrt(2.0);
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHidden, accuracy);
    }
    // Output layer (IDENTITY) weights idx 9-11: unaffected.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_GlorotBengio_OnlyScalesTanhLayers)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_GLOROTBENGIO);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Hidden layer (TANH) weights idx 0-5: *= sqrt(6/(nIn=2 + nOut=3)).
    double const expectedHidden = sqrt(6.0 / 5.0);
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHidden, accuracy);
    }
    // Output layer (IDENTITY) weights idx 9-11: unaffected.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_PreconditionOutput)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    double const mean = 2.0;
    double const sigma = 3.0;
    nn.modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT, mean, sigma);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Output weights idx 9-11: *= sigma.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - sigma, accuracy);
    }
    // Output bias idx 12: += mean.
    BOOST_REQUIRE_SMALL(out.at(12) - (1.0 + mean), accuracy);
    // Hidden layer untouched.
    for (int i = 0; i < 9; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_NguyenWidrow_AllOutputWeightsHalved)
{
    // Nguyen-Widrow initialization is documented (and, by its formula,
    // intended) to halve every weight feeding into the output neuron(s).
    // With more than one neuron in the last hidden layer, this currently
    // only holds for the very first incoming weight -- see the
    // corresponding fix in NeuralNetwork.cpp (loop bound
    // outputLayer->numNeurons should be outputLayer->numNeuronsPrevLayer).
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_NGUYENWIDROW);

    vector<double> out(n);
    nn.getConnections(out.data());

    // Hidden layer (idx 0-5, TANH, numNeuronsPrevLayer=2, numNeurons=3):
    // beta = 0.7 * 3^(1/2); each neuron's 2 weights start at 1.0, so
    // ||w|| = sqrt(2) before scaling.
    double const beta = 0.7 * pow(3.0, 1.0 / 2.0);
    double const expectedHiddenWeight = 1.0 * (beta / sqrt(2.0)) * 2.0;
    double const expectedHiddenBias   = 1.0 * beta * 2.0;
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHiddenWeight, accuracyNumeric);
    }
    for (int i = 6; i < 9; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHiddenBias, accuracyNumeric);
    }

    // Output layer weights idx 9-11 (3 hidden neurons -> 1 output neuron):
    // every one of them should be halved, not just the first.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 0.5, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(InitializeConnectionsRandomUniform_RangeAndReproducibility)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};

    NeuralNetwork nn1(numLayers, numNeuronsPerLayer, af);
    NeuralNetwork nn2(numLayers, numNeuronsPerLayer, af);

    nn1.initializeConnectionsRandomUniform(2024);
    nn2.initializeConnectionsRandomUniform(2024);

    int const n = nn1.getNumConnections();
    vector<double> c1(n);
    vector<double> c2(n);
    nn1.getConnections(c1.data());
    nn2.getConnections(c2.data());

    for (int i = 0; i < n; ++i)
    {
        BOOST_REQUIRE(c1.at(i) >= -1.0 && c1.at(i) <= 1.0);
        // Same seed must reproduce the same sequence.
        BOOST_REQUIRE_SMALL(c1.at(i) - c2.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(NeuronStatistics_AccumulateAndReset)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    double input1[2] = {1.0, 3.0};
    nn.setInput(input1);
    double input2[2] = {-2.0, 5.0};
    nn.setInput(input2);

    int const numNeurons = nn.getNumNeurons();
    vector<long> count(numNeurons);
    vector<double> minV(numNeurons);
    vector<double> maxV(numNeurons);
    vector<double> sumV(numNeurons);
    vector<double> sum2V(numNeurons);
    nn.getNeuronStatistics(count.data(), minV.data(), maxV.data(),
                           sumV.data(), sum2V.data());

    // Input neuron 0 saw values 1.0 and -2.0.
    BOOST_REQUIRE_EQUAL(count.at(0), 2);
    BOOST_REQUIRE_SMALL(minV.at(0) - (-2.0), accuracy);
    BOOST_REQUIRE_SMALL(maxV.at(0) - 1.0, accuracy);
    BOOST_REQUIRE_SMALL(sumV.at(0) - (1.0 + (-2.0)), accuracy);
    BOOST_REQUIRE_SMALL(sum2V.at(0) - (1.0 * 1.0 + 2.0 * 2.0), accuracy);
    // Input neuron 1 saw values 3.0 and 5.0.
    BOOST_REQUIRE_EQUAL(count.at(1), 2);
    BOOST_REQUIRE_SMALL(minV.at(1) - 3.0, accuracy);
    BOOST_REQUIRE_SMALL(maxV.at(1) - 5.0, accuracy);

    nn.resetNeuronStatistics();
    nn.getNeuronStatistics(count.data(), minV.data(), maxV.data(),
                           sumV.data(), sum2V.data());
    for (int i = 0; i < numNeurons; ++i)
    {
        BOOST_REQUIRE_EQUAL(count.at(i), 0);
        BOOST_REQUIRE_SMALL(sumV.at(i), accuracy);
        BOOST_REQUIRE_SMALL(sum2V.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ActivationFromString_KnownLetters_CorrectValues)
{
    BOOST_REQUIRE_EQUAL(activationFromString("l"), NeuralNetwork::AF_IDENTITY);
    BOOST_REQUIRE_EQUAL(activationFromString("t"), NeuralNetwork::AF_TANH);
    BOOST_REQUIRE_EQUAL(activationFromString("s"), NeuralNetwork::AF_LOGISTIC);
    BOOST_REQUIRE_EQUAL(activationFromString("p"), NeuralNetwork::AF_SOFTPLUS);
    BOOST_REQUIRE_EQUAL(activationFromString("r"), NeuralNetwork::AF_RELU);
    BOOST_REQUIRE_EQUAL(activationFromString("g"), NeuralNetwork::AF_GAUSSIAN);
    BOOST_REQUIRE_EQUAL(activationFromString("c"), NeuralNetwork::AF_COS);
    BOOST_REQUIRE_EQUAL(activationFromString("S"), NeuralNetwork::AF_REVLOGISTIC);
    BOOST_REQUIRE_EQUAL(activationFromString("e"), NeuralNetwork::AF_EXP);
    BOOST_REQUIRE_EQUAL(activationFromString("h"), NeuralNetwork::AF_HARMONIC);
}

BOOST_AUTO_TEST_CASE(ActivationFromString_UnknownLetter_Throws)
{
    BOOST_REQUIRE_THROW(activationFromString("z"), runtime_error);
}

BOOST_AUTO_TEST_CASE(Info_NonEmpty)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    vector<string> const info = nn.info();
    BOOST_REQUIRE(!info.empty());

    BOOST_REQUIRE(nn.getMemoryUsage() > 0);
}

BOOST_FIXTURE_TEST_CASE(WriteConnections_ProducesNonEmptyFile, FixtureRepairDir)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    ofstream file("weights.dat");
    nn.writeConnections(file);
    file.close();

    BOOST_REQUIRE(bfs::exists("weights.dat"));
    BOOST_REQUIRE(bfs::file_size("weights.dat") > 0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
