#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Log
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Log.h"
#include <cstdio>  // FILE, fopen, fclose
#include <fstream> // std::ofstream, std::ifstream
#include <iterator> // std::istreambuf_iterator
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Defaults)
{
    Log log;
    BOOST_REQUIRE(log.writeToStdout);
    BOOST_REQUIRE(!log.silent);
    BOOST_REQUIRE_EQUAL(log.getLog().size(), 0);
}

BOOST_AUTO_TEST_CASE(AddLogEntry_StoresInMemory)
{
    Log log;
    log.writeToStdout = false;
    log.addLogEntry("entry1");

    vector<string> const mem = log.getLog();
    BOOST_REQUIRE_EQUAL(mem.size(), 1);
    BOOST_REQUIRE_EQUAL(mem.at(0), "entry1");
}

BOOST_AUTO_TEST_CASE(AddMultipleLogEntries_AppendsAllInOrder)
{
    Log log;
    log.writeToStdout = false;
    log.addMultipleLogEntries({"a", "b", "c"});

    vector<string> const mem = log.getLog();
    BOOST_REQUIRE_EQUAL(mem.size(), 3);
    BOOST_REQUIRE_EQUAL(mem.at(0), "a");
    BOOST_REQUIRE_EQUAL(mem.at(1), "b");
    BOOST_REQUIRE_EQUAL(mem.at(2), "c");
}

BOOST_AUTO_TEST_CASE(OperatorShiftString_EquivalentToAddLogEntry_AndChains)
{
    Log log;
    log.writeToStdout = false;
    log << "a" << "b";

    vector<string> const mem = log.getLog();
    BOOST_REQUIRE_EQUAL(mem.size(), 2);
    BOOST_REQUIRE_EQUAL(mem.at(0), "a");
    BOOST_REQUIRE_EQUAL(mem.at(1), "b");
}

BOOST_AUTO_TEST_CASE(OperatorShiftVector_EquivalentToAddMultipleLogEntries)
{
    Log log;
    log.writeToStdout = false;
    vector<string> const entries = {"x", "y"};
    log << entries;

    vector<string> const mem = log.getLog();
    BOOST_REQUIRE_EQUAL(mem.size(), 2);
    BOOST_REQUIRE_EQUAL(mem.at(0), "x");
    BOOST_REQUIRE_EQUAL(mem.at(1), "y");
}

BOOST_AUTO_TEST_CASE(Silent_SuppressesStorage)
{
    Log log;
    log.writeToStdout = false;
    log.silent = true;
    log.addLogEntry("hidden");

    BOOST_REQUIRE_EQUAL(log.getLog().size(), 0);
}

BOOST_FIXTURE_TEST_CASE(RegisterStreamPointer_WritesEntryToStream,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Log log;
    log.writeToStdout = false;

    ofstream logFile("log.txt");
    log.registerStreamPointer(&logFile);
    log.addLogEntry("hello\n");
    logFile.flush();

    ifstream check("log.txt");
    string const content((istreambuf_iterator<char>(check)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE_EQUAL(content, "hello\n");
    // Memory still records the entry independently of the stream sink.
    BOOST_REQUIRE_EQUAL(log.getLog().size(), 1);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_CASE(RegisterStreamPointer_UnopenedStream_SkippedWithoutCrash)
{
    Log log;
    log.writeToStdout = false;

    ofstream unopened; // Never opened -> is_open() == false.
    log.registerStreamPointer(&unopened);

    BOOST_REQUIRE_NO_THROW(log.addLogEntry("still recorded"));
    BOOST_REQUIRE_EQUAL(log.getLog().size(), 1);
}

BOOST_FIXTURE_TEST_CASE(RegisterCFilePointer_WritesEntryToFile,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Log log;
    log.writeToStdout = false;

    FILE* filePointer = fopen("log.txt", "w");
    log.registerCFilePointer(&filePointer);
    log.addLogEntry("hello\n");

    ifstream check("log.txt");
    string const content((istreambuf_iterator<char>(check)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE_EQUAL(content, "hello\n");

    fclose(filePointer);
    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_CASE(RegisterCFilePointer_NullPointer_SkippedWithoutCrash)
{
    Log log;
    log.writeToStdout = false;

    FILE* nullFilePointer = nullptr;
    log.registerCFilePointer(&nullFilePointer);

    BOOST_REQUIRE_NO_THROW(log.addLogEntry("still recorded"));
    BOOST_REQUIRE_EQUAL(log.getLog().size(), 1);
}

BOOST_AUTO_TEST_SUITE_END()
