#ifndef FIXTURES_H
#define FIXTURES_H

#include "Atom.h"
#include "ElementMap.h"
#include "Structure.h"

using namespace std;
using namespace nnp;

struct FixtureThreeAtomsMono
{
    FixtureThreeAtomsMono()
    {
        em.registerElements("H");
        s.setElementMap(em);

        // Atom 1.
        s.atoms.push_back(Atom());
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 2.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 1.0;
        s.atoms.back().r[1] = 2.0;
        s.atoms.back().r[2] = 1.0;
        s.atoms.back().index = 1;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 3.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 0.0;
        s.atoms.back().r[1] = 3.0;
        s.atoms.back().r[2] = -1.0;
        s.atoms.back().index = 2;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Calculate neighbor list.
        s.calculateNeighborList(10.0);
    }
    ~FixtureThreeAtomsMono() {};

    ElementMap em;
    Structure s;
};

struct FixtureFourAtomsMixed
{
    FixtureFourAtomsMixed()
    {
        em.registerElements("S Cu U");
        s.setElementMap(em);

        // Atom 1.
        s.atoms.push_back(Atom());
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 2.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 1.0;
        s.atoms.back().r[1] = 2.0;
        s.atoms.back().r[2] = 1.0;
        s.atoms.back().index = 1;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 3.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 0.0;
        s.atoms.back().r[1] = 3.0;
        s.atoms.back().r[2] = -1.0;
        s.atoms.back().index = 2;
        s.atoms.back().element = 1;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[1]++;

        // Atom 4.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = -2.0;
        s.atoms.back().r[1] = 1.0;
        s.atoms.back().r[2] = -0.5;
        s.atoms.back().index = 3;
        s.atoms.back().element = 2;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[2]++;

        // Calculate neighbor list.
        s.calculateNeighborList(10.0);
    };
    ~FixtureFourAtomsMixed() {};

    ElementMap em;
    Structure s;
};

vector<size_t> const typesMono = {2, 2, 3, 3, 9, 9, 28, 28, 28, 29, 29 ,29, 89, 89, 89, 99, 99, 99,
                                  280, 280, 280, 890, 890, 890, 990, 990, 990};
vector<string> const setupLinesMono = {"H 2 H 0.001 0.0 10.0",
                                       "H 2 H 0.030 1.0 10.0",
                                       "H 3 H H 0.001  1.0 1.0 10.0 0.0",
                                       "H 3 H H 0.030 -1.0 4.0 10.0 2.0",
                                       "H 9 H H 0.001  1.0 1.0 10.0 0.0",
                                       "H 9 H H 0.031 -1.0 4.0 10.0 2.0",
                                       "H 28 H    -10.0 10.0",
                                       "H 28 H      0.0  8.0",
                                       "H 28 H      1.0  3.0",
                                       "H 29 H H 0.001   0.0 180.0 10.0 0.0",
                                       "H 29 H H 0.031  10.0  60.0 10.0 0.0",
                                       "H 29 H H 0.031  40.0 120.0 10.0 1.0",
                                       "H 89 H H  -10.0  0.0 180.0 10.0",
                                       "H 89 H H    0.0 10.0  60.0 10.0",
                                       "H 89 H H    1.0 40.0 120.0 10.0",
                                       "H 99 H H  -10.0  0.0 180.0 10.0",
                                       "H 99 H H    0.0 10.0  60.0 10.0",
                                       "H 99 H H    1.0 40.0 120.0 10.0",
                                       "H 280 H    -10.0 10.0",
                                       "H 280 H      0.0  8.0",
                                       "H 280 H      1.0  3.0",
                                       "H 890 H H  -10.0  0.0 180.0 10.0",
                                       "H 890 H H    0.0 10.0  60.0 10.0",
                                       "H 890 H H    1.0 40.0 120.0 10.0",
                                       "H 990 H H  -10.0  0.0 180.0 10.0",
                                       "H 990 H H    0.0 10.0  60.0 10.0",
                                       "H 990 H H    1.0 40.0 120.0 10.0"};
vector<double> const valuesMono = {4.6578975865793892E-01,
                                   4.2613074319879601E-01,
                                   2.2781144245150042E-02,
                                   2.6506732726623567E-05,
                                   8.8187604914213763E-02,
                                   1.0245670290254523E-04, 
                                   1.71653828561074920,
                                   1.63829178699555840,
                                   0.59406437434580717, 
                                   0.032112222812294255,
                                   0.010995454898531647,
                                   0.0043361052372807912,
                                   0.440241957350979370,
                                   0.117549886737894540,
                                   0.008811770360680239,
                                   0.39698365829232357,
                                   0.056548996092037257,
                                   0.0017053618495084467,
                                   1.0690455910728218,
                                   1.0,
                                   0.16715396673145347,
                                   0.16576440370335888,
                                   0.0098147698371699576,
                                   7.8782625391130029e-05,
                                   0.10438665096250159,
                                   0.00091523881378179322,
                                   7.4866566532293829e-07};

vector<size_t> const typesMixed = {2, 2, 3, 3, 3, 12, 12, 13, 13}; 
vector<string> const setupLinesMixed = {"S 2 Cu 0.030 1.0 10.0",
                                        "S 2  U 0.030 1.0 10.0",
                                        "S 3  S Cu 0.001  1.0 1.0 10.0 0.0",
                                        "S 3  S  U 0.001  1.0 1.0 10.0 0.0",
                                        "S 3  Cu U 0.001  1.0 1.0 10.0 0.0",
                                        "S 12 0.001 0.0 10.0",
                                        "S 12 0.030 1.0 10.0",
                                        "S 13 0.001 0.0  1.0 1.0 10.0",
                                        "S 13 0.001 1.0 -1.0 4.0 10.0"};
vector<double> const valuesMixed = {1.8212508087363616E-01,
                                    2.5814603379946083E-01,
                                    2.2781144245150042E-02,
                                    1.1723866787654575E-02,
                                    1.8921487310992163E-02,
                                    3.4986591461381153E+01,
                                    3.2935153052088403E+01,
                                    7.8310510986904234E+01,
                                    3.6978004634143264E+00};

#endif
