#ifndef NNP_TEST_H
#define NNP_TEST_H

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
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
    boost::asio::io_service svc;\
    std::future<std::vector<char> > outStream;\
    bproc::child c(example.command + " " + example.args,\
                   bproc::std_out > outStream,\
                   svc);\
    svc.run();\
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
    //BOOST_REQUIRE_EQUAL(result, 0);
    //int result = bproc::system(example.command + " " + example.args,
    //                           bproc::std_out > outStream);
    //bproc::ipstream outStream;
