#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE utility
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "utility.h"
#include <cmath>     // std::isnan
#include <fstream>   // std::ofstream, std::ifstream
#include <limits>    // std::numeric_limits
#include <map>       // std::map
#include <stdexcept> // std::range_error, std::runtime_error
#include <string>    // std::string
#include <unordered_map> // std::unordered_map
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Split_DefaultDelimiter_SplitsOnSpaces)
{
    vector<string> const parts = split("a b  c");
    // A run of two consecutive delimiters produces an empty part.
    BOOST_REQUIRE_EQUAL(parts.size(), 4);
    BOOST_REQUIRE_EQUAL(parts.at(0), "a");
    BOOST_REQUIRE_EQUAL(parts.at(1), "b");
    BOOST_REQUIRE_EQUAL(parts.at(2), "");
    BOOST_REQUIRE_EQUAL(parts.at(3), "c");
}

BOOST_AUTO_TEST_CASE(Split_CustomDelimiter_Works)
{
    vector<string> const parts = split("a,b,c", ',');
    BOOST_REQUIRE_EQUAL(parts.size(), 3);
    BOOST_REQUIRE_EQUAL(parts.at(0), "a");
    BOOST_REQUIRE_EQUAL(parts.at(1), "b");
    BOOST_REQUIRE_EQUAL(parts.at(2), "c");
}

BOOST_AUTO_TEST_CASE(Split_EmptyString_ReturnsEmptyVector)
{
    BOOST_REQUIRE_EQUAL(split("").size(), 0);
}

BOOST_AUTO_TEST_CASE(Split_NoDelimiterPresent_ReturnsWholeString)
{
    vector<string> const parts = split("abc", ',');
    BOOST_REQUIRE_EQUAL(parts.size(), 1);
    BOOST_REQUIRE_EQUAL(parts.at(0), "abc");
}

BOOST_AUTO_TEST_CASE(Trim_LeadingAndTrailingWhitespace_Removed)
{
    BOOST_REQUIRE_EQUAL(trim("  \t hello \t "), "hello");
}

BOOST_AUTO_TEST_CASE(Trim_AllWhitespace_ReturnsEmpty)
{
    BOOST_REQUIRE_EQUAL(trim("  \t  "), "");
}

BOOST_AUTO_TEST_CASE(Trim_NoWhitespace_Unchanged)
{
    BOOST_REQUIRE_EQUAL(trim("hello"), "hello");
}

BOOST_AUTO_TEST_CASE(Trim_CustomWhitespaceSet_Works)
{
    BOOST_REQUIRE_EQUAL(trim("xxhelloxx", "x"), "hello");
}

BOOST_AUTO_TEST_CASE(Reduce_CollapsesInternalWhitespaceRuns)
{
    BOOST_REQUIRE_EQUAL(reduce("  a   b\tc  "), "a b c");
}

BOOST_AUTO_TEST_CASE(Reduce_CustomFill_Works)
{
    BOOST_REQUIRE_EQUAL(reduce("a   b", " \t", "-"), "a-b");
}

BOOST_AUTO_TEST_CASE(Pad_RightPad_AppendsFillToDesiredLength)
{
    BOOST_REQUIRE_EQUAL(pad("ab", 5), "ab   ");
}

BOOST_AUTO_TEST_CASE(Pad_LeftPad_PrependsFillToDesiredLength)
{
    BOOST_REQUIRE_EQUAL(pad("ab", 5, '0', false), "000ab");
}

BOOST_AUTO_TEST_CASE(Pad_AlreadyLongEnough_ReturnsUnchanged)
{
    BOOST_REQUIRE_EQUAL(pad("abcdef", 3), "abcdef");
}

BOOST_AUTO_TEST_CASE(Pad_CustomFillChar_Works)
{
    BOOST_REQUIRE_EQUAL(pad("7", 3, '0', false), "007");
}

BOOST_AUTO_TEST_CASE(Strpr_FormatsLikePrintf)
{
    BOOST_REQUIRE_EQUAL(strpr("%d-%s-%.2f", 3, "x", 1.5), "3-x-1.50");
}

BOOST_AUTO_TEST_CASE(Cap_CapitalizesFirstLetter)
{
    BOOST_REQUIRE_EQUAL(cap("hello"), "Hello");
}

BOOST_AUTO_TEST_CASE(Cap_AlreadyCapitalized_Unchanged)
{
    BOOST_REQUIRE_EQUAL(cap("Hello"), "Hello");
}

BOOST_AUTO_TEST_CASE(CreateFileHeader_MismatchedColumnSizes_Throws)
{
    vector<string> const title = {"Title"};
    vector<size_t> const colSize = {10, 10};
    vector<string> const colName = {"a"}; // Mismatched length vs colSize.
    vector<string> const colInfo = {"first", "second"};

    BOOST_REQUIRE_THROW(createFileHeader(title, colSize, colName, colInfo),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CreateFileHeader_WellFormed_ContainsExpectedContent)
{
    vector<string> const title = {"My header"};
    vector<size_t> const colSize = {10, 10};
    vector<string> const colName = {"colA", "colB"};
    vector<string> const colInfo = {"first column", "second column"};

    vector<string> const h = createFileHeader(title, colSize, colName,
                                              colInfo);

    string joined;
    for (auto const& line : h) joined += line;

    BOOST_REQUIRE(joined.find("My header") != string::npos);
    BOOST_REQUIRE(joined.find("colA") != string::npos);
    BOOST_REQUIRE(joined.find("colB") != string::npos);
    BOOST_REQUIRE(joined.find("first column") != string::npos);
    BOOST_REQUIRE(joined.find("second column") != string::npos);
    // Every line should start with the default comment character.
    for (auto const& line : h)
    {
        BOOST_REQUIRE_EQUAL(line.at(0), '#');
    }
}

BOOST_AUTO_TEST_CASE(CreateFileHeader_CustomCommentChar_Used)
{
    vector<string> const title = {"T"};
    vector<size_t> const colSize = {5};
    vector<string> const colName = {"a"};
    vector<string> const colInfo = {"info"};

    vector<string> const h = createFileHeader(title, colSize, colName,
                                              colInfo, '!');
    BOOST_REQUIRE_EQUAL(h.front().at(0), '!');
}

BOOST_FIXTURE_TEST_CASE(AppendLinesToFile_Ofstream_WritesAllLines,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    vector<string> const lines = {"line1\n", "line2\n"};
    ofstream file("out.txt");
    appendLinesToFile(file, lines);
    file.close();

    ifstream check("out.txt");
    string content((istreambuf_iterator<char>(check)),
                   istreambuf_iterator<char>());
    BOOST_REQUIRE_EQUAL(content, "line1\nline2\n");

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(AppendLinesToFile_FilePointer_WritesAllLines,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    vector<string> const lines = {"line1\n", "line2\n"};
    FILE* file = fopen("out.txt", "w");
    appendLinesToFile(file, lines);
    fclose(file);

    ifstream check("out.txt");
    string content((istreambuf_iterator<char>(check)),
                   istreambuf_iterator<char>());
    BOOST_REQUIRE_EQUAL(content, "line1\nline2\n");

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(ReadColumnsFromFile_SkipsCommentsAndBlankLines,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    ofstream data("data.txt");
    data << "# comment line\n";
    data << "1.0 2.0 3.0\n";
    data << "\n";
    data << "4.0 5.0 6.0\n";
    data.close();

    map<size_t, vector<double>> const result =
        readColumnsFromFile("data.txt", {0, 2});

    BOOST_REQUIRE_EQUAL(result.size(), 2);
    BOOST_REQUIRE_EQUAL(result.at(0).size(), 2);
    BOOST_REQUIRE_EQUAL(result.at(0).at(0), 1.0);
    BOOST_REQUIRE_EQUAL(result.at(0).at(1), 4.0);
    BOOST_REQUIRE_EQUAL(result.at(2).at(0), 3.0);
    BOOST_REQUIRE_EQUAL(result.at(2).at(1), 6.0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(ReadColumnsFromFile_MissingColumn_FillsNaN,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    ofstream data("data.txt");
    data << "1.0 2.0\n"; // Only 2 columns present.
    data.close();

    map<size_t, vector<double>> const result =
        readColumnsFromFile("data.txt", {0, 5});

    BOOST_REQUIRE_EQUAL(result.at(0).at(0), 1.0);
    BOOST_REQUIRE(isnan(result.at(5).at(0)));

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_CASE(ReadColumnsFromFile_NonexistentFile_Throws)
{
    BOOST_REQUIRE_THROW(readColumnsFromFile("does_not_exist.txt", {0}),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(SafeFind_ExistingKey_ReturnsValue)
{
    map<string, int> const m = {{"a", 1}, {"b", 2}};
    BOOST_REQUIRE_EQUAL(safeFind(m, string("b")), 2);
}

BOOST_AUTO_TEST_CASE(SafeFind_MissingKey_Throws)
{
    map<string, int> const m = {{"a", 1}};
    BOOST_REQUIRE_THROW(safeFind(m, string("z")), range_error);
}

BOOST_AUTO_TEST_CASE(VectorContains_FindsPresentValue)
{
    vector<int> const v = {1, 2, 3};
    BOOST_REQUIRE(vectorContains(v, 2));
    BOOST_REQUIRE(!vectorContains(v, 4));
}

BOOST_AUTO_TEST_CASE(UnorderedMapContainsKey_DetectsPresenceCorrectly)
{
    unordered_map<int, string> const m = {{1, "a"}, {2, "b"}};
    BOOST_REQUIRE(unorderedMapContainsKey(m, 1));
    BOOST_REQUIRE(!unorderedMapContainsKey(m, 3));
}

BOOST_AUTO_TEST_CASE(ComparePointerTargets_ComparesDereferencedValues)
{
    int a = 1;
    int b = 2;
    BOOST_REQUIRE(comparePointerTargets(&a, &b));
    BOOST_REQUIRE(!comparePointerTargets(&b, &a));
}

BOOST_AUTO_TEST_CASE(PowInt_PositiveExponent_MatchesDirectComputation)
{
    BOOST_REQUIRE_EQUAL(pow_int(2.0, 10), 1024.0);
    BOOST_REQUIRE_EQUAL(pow_int(-2.0, 3), -8.0);
}

BOOST_AUTO_TEST_CASE(PowInt_ZeroExponent_ReturnsOne)
{
    BOOST_REQUIRE_EQUAL(pow_int(5.0, 0), 1.0);
}

BOOST_AUTO_TEST_CASE(PowInt_NegativeExponent_ReturnsInverse)
{
    BOOST_REQUIRE_EQUAL(pow_int(2.0, -2), 0.25);
}

BOOST_AUTO_TEST_SUITE_END()
