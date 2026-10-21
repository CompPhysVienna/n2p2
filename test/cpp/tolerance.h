#ifndef TOLERANCE_H
#define TOLERANCE_H

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::max
#include <cmath>     // std::fabs
#include <iomanip>   // std::setprecision

// Tolerance for comparing a computed value against a stored reference value.
//
// A value passes if |value - reference| <= max(absolute, relative * |reference|).
// The relative part defaults to 0: the tests keep their historical, purely
// absolute tolerances.
struct Tolerance
{
    double absolute;
    double relative;

    Tolerance(double absolute, double relative = 0.0) :
        absolute(absolute),
        relative(relative)
    {
    }

    double bound(double reference) const
    {
        return std::max(absolute, relative * std::fabs(reference));
    }
};

// Like BOOST_REQUIRE_SMALL(value - reference, ...), but with a Tolerance and
// a message stating reference, value, deviation and allowed bound.
#define REQUIRE_CLOSE_TO_REFERENCE(reference, value, tolerance)               \
    BOOST_REQUIRE_MESSAGE(                                                     \
        std::fabs((value) - (reference)) <= (tolerance).bound(reference),      \
        std::setprecision(17) << "deviation |" #value " - " #reference "| = " \
        << std::fabs((value) - (reference)) << " exceeds "                     \
        << (tolerance).bound(reference) << " (reference "                      \
        << (reference) << ", value " << (value) << ")")

#endif
