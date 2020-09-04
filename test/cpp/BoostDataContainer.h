#ifndef BOOSTDATACONTAINER_H
#define BOOSTDATACONTAINER_H

#include <vector> // std::vector

template <class T> struct BoostDataContainer
{
    BoostDataContainer()
    {
        setup();
    }

    void setup()
    {
        return;
    }

    std::vector<T> examples;
};

#endif
