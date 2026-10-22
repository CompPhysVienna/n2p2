#ifndef FILEHELPERS_H
#define FILEHELPERS_H

#include <fstream>   // std::ifstream, std::ofstream
#include <iostream>
#include <boost/filesystem.hpp>
#include <sstream>   // std::ostringstream
#include <string>

namespace bfs = boost::filesystem;

// Replaces the first occurrence of `from` with `to` in the file at `path`.
// Returns false if the file could not be read or `from` was not found.
bool replaceInFile(bfs::path const& path,
                   std::string const& from,
                   std::string const& to)
{
    std::ifstream in(path.string());
    if (!in.is_open()) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    in.close();

    std::string content = buffer.str();
    std::size_t const pos = content.find(from);
    if (pos == std::string::npos) return false;
    content.replace(pos, from.size(), to);

    std::ofstream out(path.string(), std::ios::trunc);
    if (!out.is_open()) return false;
    out << content;
    return true;
}

bool copy_directory_recursively(bfs::path const& src, bfs::path const& dest)
{
    try
    {
        if(!bfs::exists(src) || !bfs::is_directory(src))
        {
            std::cerr << "Source directory " << src.string()
                      << " does not exist or is not a directory.\n";
            return false;
        }
        if(bfs::exists(dest))
        {
            std::cerr << "Destination directory " << dest.string()
                      << " already exists.\n";
            return false;
        }
        if(!bfs::create_directory(dest))
        {
            std::cerr << "Unable to create destination directory"
                      << dest.string() << '\n';
            return false;
        }
    }
    catch(bfs::filesystem_error const & e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
    for(bfs::directory_iterator file(src);
        file != bfs::directory_iterator(); ++file)
    {
        try
        {
            bfs::path current(file->path());
            if(bfs::is_directory(current))
            {
                if(!copy_directory_recursively(current,
                                               dest / current.filename()))
                {
                    return false;
                }
            }
            else
            {
                bfs::copy_file(current, dest / current.filename());
            }
        }
        catch(bfs::filesystem_error const & e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    return true;
}

struct FixtureRepairDir
{
    FixtureRepairDir()
    {
        std::string p = bfs::current_path().string();
        if (p.substr(p.size() - 4) == "test")
        {
            bfs::current_path("..");
            bfs::remove_all("test");
        }
        else if (bfs::exists("test"))
        {
            bfs::remove_all("test");
        }
    }

};

#endif
