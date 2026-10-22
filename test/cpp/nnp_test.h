#ifndef NNP_TEST_H
#define NNP_TEST_H

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include <cstdlib>  // std::system
#include <iostream>
#include <string>   // std::string
#include "fileHelpers.h"

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

// Runs the tool of each example in a copy of its directory, then calls
// nnpToolTestBody(), which checks the files the tool wrote. The tool's
// standard output goes to stdout.txt there. std::system() is used instead
// of Boost.Process, whose interface changed in recent Boost versions
// (Process v2 since 1.86, Asio's io_service removed).
#define NNP_TOOL_TEST_CASE()\
\
BOOST_AUTO_TEST_SUITE(RegressionTests)\
\
BOOST_DATA_TEST_CASE_F(FixtureRepairDir,\
                       Execute_CorrectOutputFiles,\
                       bdata::make(container.examples),\
                       example)\
{\
    BOOST_REQUIRE(copy_directory_recursively(example.pathData, "test"));\
    bfs::current_path("test");\
\
    std::string const command = example.command + " " + example.args\
                              + " > stdout.txt";\
    int const status = std::system(command.c_str());\
    (void)status;\
\
    BOOST_TEST_INFO(example.description);\
\
    nnpToolTestBody(example);\
\
    bfs::current_path("..");\
    bfs::remove_all("test");\
}\
\
BOOST_AUTO_TEST_SUITE_END()

#endif
