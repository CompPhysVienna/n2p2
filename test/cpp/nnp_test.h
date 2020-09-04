#ifndef NNP_TEST_H
#define NNP_TEST_H

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include "fileHelpers.h"

namespace bdata = boost::unit_test::data;
namespace bproc = boost::process;
namespace bfs = boost::filesystem;

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
    std::future<std::vector<char>> outStream;\
    boost::asio::io_service io;\
    bproc::child c(example.command + " " + example.args,\
                   bproc::std_in.close(),\
                   bproc::std_out > outStream,\
                   io);\
    io.run();\
    c.wait();\
    int result = c.exit_code();\
\
    BOOST_TEST_INFO(example.description);\
    BOOST_REQUIRE_EQUAL(result, 0);\
\
    nnpToolTestBody(example);\
\
    bfs::current_path("..");\
    bfs::remove_all("test");\
}\
\
BOOST_AUTO_TEST_SUITE_END()

#endif
