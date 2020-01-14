#ifndef FILEHELPERS_H
#define FILEHELPERS_H

#include <iostream>
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

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

#endif
