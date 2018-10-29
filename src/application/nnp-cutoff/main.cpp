// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "CutoffFunction.h"
#include "Log.h"
#include "Stopwatch.h"
#include "utility.h"
#include <iostream> // std::cout
#include <fstream>  // std::ofstream

using namespace std;
using namespace nnp;

CutoffFunction fc;
Log log;
ofstream logFile;
Stopwatch swf;
Stopwatch swdf;
Stopwatch swfdf;
size_t p = 100000000;
double d = 0.99999 / p;
double tbf = 0.0;
double tbdf = 0.0;
double tbfdf = 0.0;
double tlf = 0.0;
double tldf = 0.0;
double tlfdf = 0.0;

void runTest(bool write)
{
    swf.reset();
    swf.start();
    for (size_t i = 0; i < p; ++i)
    {
        __asm__ __volatile__("");
    }
    tlf = swf.stop();
    if (write) log << strpr(" %10.2E %6.1f       ", tlf, tlf / tlf);

    swdf.reset();
    swdf.start();
    for (size_t i = 0; i < p; ++i)
    {
        __asm__ __volatile__("");
    }
    tldf = swdf.stop();
    if (write) log << strpr(" %10.2E %6.1f       ", tldf, tldf / tldf);

    swfdf.reset();
    swfdf.start();
    for (size_t i = 0; i < p; ++i)
    {
        __asm__ __volatile__("");
    }
    tlfdf = swfdf.stop();
    if (write) log << strpr(" %10.2E %6.1f        %10.2E %9.1f\n",
                            tlfdf,
                            tlfdf / tlfdf,
                            tlfdf - (tlf + tldf),
                            100.0 * tlfdf / (tlf + tldf));
}

void runTest(CutoffFunction::CutoffType cutoffType)
{
    fc.setCutoffType(cutoffType);

    swf.reset();
    swf.start();
    for (size_t i = 0; i < p; ++i)
    {
        fc.f(i * d);
    }
    double tf = swf.stop();
    if (cutoffType == CutoffFunction::CT_HARD) tbf = tf;
    log << strpr(" %10.2E %6.1f %6.1f", tf, tf / tlf, tf / tbf);

    swdf.reset();
    swdf.start();
    for (size_t i = 0; i < p; ++i)
    {
        fc.df(i * d);
    }
    double tdf = swdf.stop();
    if (cutoffType == CutoffFunction::CT_HARD) tbdf = tdf;
    log << strpr(" %10.2E %6.1f %6.1f", tdf, tdf / tldf, tdf / tbdf);

    swfdf.reset();
    swfdf.start();
    for (size_t i = 0; i < p; ++i)
    {
        double f;
        double df;
        fc.fdf(i * d, f, df);
    }
    double tfdf = swfdf.stop();
    if (cutoffType == CutoffFunction::CT_HARD) tbfdf = tfdf;

    log << strpr(" %10.2E %6.1f %6.1f %10.2E %9.1f\n",
                 tfdf,
                 tfdf / tlfdf,
                 tfdf / tbfdf,
                 tfdf - (tf + tdf),
                 100.0 * tfdf / (tf + tdf));
}

int main()
{
    logFile.open("nnp-cutoff.log");
    log.registerStreamPointer(&logFile);

    log << "-------------------------------------------------------------------------------------------------------------\n";
    log << "Speed test tool for cutoff functions:\n";
    log << "-------------------------------------------------------------------------------------------------------------\n";
    log << "Column f     : Time for calling f   (no derivatives  ).\n";
    log << "Column df    : Time for calling df  (only derivatives).\n";
    log << "Column fdf   : Time for calling fdf (f and df at once).\n";
    log << "Column compL : Time compared to LOOP ONLY.\n";
    log << "Column compH : Time compared to CT_HARD.\n";
    log << "Column diff  : Time difference between calling f + df (separately) and fdf.\n";
    log << "Column ratio : Ratio time(fdf) / (time(f) + time(df)) in %.\n";
    log << "-------------------------------------------------------------------------------------------------------------\n";
    log << "CutoffType :       f [s]  compL  compH     df [s]  compL  compH     fdf[s]  compL  compH   diff [s] ratio [%]\n";
    log << "-------------------------------------------------------------------------------------------------------------\n";

    // Initialize...
    runTest(false);

    log << "LOOP ONLY  : ";
    runTest(true);

    fc.setCutoffRadius(1.0);

    log << "CT_HARD    : ";
    runTest(CutoffFunction::CT_HARD);

    log << "CT_COS     : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_COS);

    log << "CT_TANHU   : ";
    runTest(CutoffFunction::CT_TANHU);

    log << "CT_TANH    : ";
    runTest(CutoffFunction::CT_TANH);

    log << "CT_EXP     : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_EXP);

    log << "CT_POLY1   : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_POLY1);

    log << "CT_POLY2   : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_POLY2);

    log << "CT_POLY3   : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_POLY3);

    log << "CT_POLY4   : ";
    fc.setCutoffParameter(0.0);
    runTest(CutoffFunction::CT_POLY4);

    logFile.close();

    return 0;
}
