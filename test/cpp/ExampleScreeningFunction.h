#ifndef EXAMPLESCREENINGFUNCTION_H
#define EXAMPLESCREENINGFUNCTION_H

#include <vector>
#include "Example.h"
#include "BoostDataContainer.h"
#include "ScreeningFunction.h"

double const radiusInner = 4.8;      // Inner limit of transition region.
double const radiusOuter = 8.0;      // Outer limit of transition region.
double const testRadius = 7.0;       // Inside transition region.
double const testRadiusBelow = 2.0;  // Below transition region.
double const testRadiusAbove = 10.0; // Above transition region.

struct ExampleScreeningFunction : public Example
{
    std::string            type;
    double                 inner;
    double                 outer;
    double                 rt;
    double                 rtbelow;
    double                 rtabove;
    double                 f;
    double                 df;
    nnp::ScreeningFunction fs;

    ExampleScreeningFunction();

    ExampleScreeningFunction(std::string name) : 
        inner  (radiusInner    ),
        outer  (radiusOuter    ),
        rt     (testRadius     ),
        rtbelow(testRadiusBelow),
        rtabove(testRadiusAbove)
    {
        this->name = name;
        this->description = std::string("ScreeningFunction example \"")
                          + this->name
                          + "\"";
    }
};

template<>
void BoostDataContainer<ExampleScreeningFunction>::setup()
{
    ExampleScreeningFunction* e = nullptr;

    examples.push_back(ExampleScreeningFunction("Cosine"));
    e = &(examples.back());
    e->type = "c";
    e->f = 7.7778511650980120E-01;
    e->df = 4.0814669151450456E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    examples.push_back(ExampleScreeningFunction("Polynomial 1"));
    e = &(examples.back());
    e->type = "p1";
    e->f = 7.6806640625000011E-01;
    e->df = 4.0283203124999994E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    examples.push_back(ExampleScreeningFunction("Polynomial 2"));
    e = &(examples.back());
    e->type = "p2";
    e->f = 8.1999397277832031E-01;
    e->df = 4.3272972106933594E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    examples.push_back(ExampleScreeningFunction("Polynomial 3"));
    e = &(examples.back());
    e->type = "p3";
    e->f = 8.5718168318271637E-01;
    e->df = 4.3385662138462061E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    examples.push_back(ExampleScreeningFunction("Polynomial 4"));
    e = &(examples.back());
    e->type = "p4";
    e->f = 8.8514509823289700E-01;
    e->df = 4.1945122575270932E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    examples.push_back(ExampleScreeningFunction("Exponential window"));
    e = &(examples.back());
    e->type = "e";
    e->f = 5.9192173471535003E-01;
    e->df = 6.3053409878824362E-01;
    e->fs.setInnerOuter(e->inner, e->outer);
    e->fs.setCoreFunction(e->type);

    return;
}


#endif
