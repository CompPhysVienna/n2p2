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

#include "Settings.h"
#include "utility.h"
#include <fstream>   // std::ifstream
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

map<string, string> const createKnownKeywordsMap()
{
    map<string, string> m;

    // Required for prediction.
    m["number_of_elements"            ] = "";
    m["elements"                      ] = "";
    m["atom_energy"                   ] = "";
    m["cutoff_type"                   ] = "";
    m["symfunction_short"             ] = "";
    m["scale_symmetry_functions"      ] = "";
    m["scale_min_short"               ] = "";
    m["scale_max_short"               ] = "";
    m["center_symmetry_functions"     ] = "";
    m["scale_symmetry_functions_sigma"] = "";
    m["global_hidden_layers_short"    ] = "";
    m["global_nodes_short"            ] = "";
    m["global_activation_short"       ] = "";
    m["normalize_nodes"               ] = "";
    m["mean_energy"                   ] = "";
    m["conv_length"                   ] = "";
    m["conv_energy"                   ] = "";

    // Training keywords.
    m["random_seed"                   ] = "";
    m["test_fraction"                 ] = "";
    m["epochs"                        ] = "";
    m["use_short_forces"              ] = "";
    m["short_energy_error_threshold"  ] = "";
    m["short_force_error_threshold"   ] = "";
    m["rmse_threshold_trials"         ] = "";
    m["short_energy_fraction"         ] = "";
    m["short_force_fraction"          ] = "";
    m["use_old_weights_short"         ] = "";
    m["weights_min"                   ] = "";
    m["weights_max"                   ] = "";
    m["nguyen_widrow_weights_short"   ] = "";
    m["precondition_weights"          ] = "";
    m["write_trainpoints"             ] = "";
    m["write_trainforces"             ] = "";
    m["write_weights_epoch"           ] = "";
    m["write_neuronstats"             ] = "";
    m["write_trainlog"                ] = "";
    m["repeated_energy_update"        ] = "";
    m["updater_type"                  ] = "";
    m["parallel_mode"                 ] = "";
    m["jacobian_mode"                 ] = "";
    m["update_strategy"               ] = "";
    m["selection_mode"                ] = "";
    m["task_batch_size_energy"        ] = "";
    m["task_batch_size_force"         ] = "";
    m["gradient_type"                 ] = "";
    m["gradient_eta"                  ] = "";
    m["gradient_adam_eta"             ] = "";
    m["gradient_adam_beta1"           ] = "";
    m["gradient_adam_beta2"           ] = "";
    m["gradient_adam_epsilon"         ] = "";
    m["kalman_type"                   ] = "";
    m["kalman_epsilon"                ] = "";
    m["kalman_eta"                    ] = "";
    m["kalman_etatau"                 ] = "";
    m["kalman_etamax"                 ] = "";
    m["kalman_q0"                     ] = "";
    m["kalman_qtau"                   ] = "";
    m["kalman_qmin"                   ] = "";
    m["kalman_lambda_short"           ] = "";
    m["kalman_nue_short"              ] = "";
    m["memorize_symfunc_results"      ] = "";
    m["force_weight"                  ] = "";

    return m;
}

map<string, string> const Settings::knownKeywords = createKnownKeywordsMap();

string Settings::operator[](string const& keyword) const
{
    return getValue(keyword);
}

void Settings::loadFile(string const& fileName)
{
    this->fileName = fileName;

    readFile();
    parseLines();
}

bool Settings::keywordExists(string const& keyword) const
{
    return (contents.find(keyword) != contents.end());
}

string Settings::getValue(string const& keyword) const
{
    if (!Settings::keywordExists(keyword))
    {
        throw std::runtime_error("ERROR: Keyword \"" + keyword
                                 + "\" not found.\n");
    }

    return contents.find(keyword)->second.first;
}

Settings::KeyRange Settings::getValues(string const& keyword) const
{
    if (!Settings::keywordExists(keyword))
    {
        throw std::runtime_error("ERROR: Keyword \"" + keyword
                                 + "\" not found.\n");
    }

    return contents.equal_range(keyword);
}

vector<string> Settings::info() const
{
    return log;
}

vector<string> Settings::getSettingsLines() const
{
    return lines;
}

void Settings::readFile()
{
    ifstream file;
    string   line;

    log.push_back(strpr("Settings file name: %s\n", fileName.c_str()));
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }

    while (getline(file, line))
    {
        lines.push_back(line);
    }

    file.close();

    log.push_back(strpr("Read %d lines.\n", lines.size()));

    return;
}

void Settings::writeSettingsFile(ofstream* const& file) const
{
    if (!file->is_open())
    {
        runtime_error("ERROR: Could not write to file.\n");
    }

    for (vector<string>::const_iterator it = lines.begin();
         it != lines.end(); ++it)
    {
        (*file) << (*it) << '\n';
    }

    return;
}

void Settings::parseLines()
{
    for (size_t i = 0; i < lines.size(); ++i)
    {
        string line = lines.at(i);

        // ignore empty and comment lines
        if (line.empty())
        {
            continue;
        }
        if (line.find('#') != string::npos)
        {
            line.erase(line.find('#'));
        }
        if (line.find('!') != string::npos)
        {
            line.erase(line.find('!'));
        }
        if (line.find_first_not_of(' ') == string::npos)
        {
            continue;
        }

        // remove leading and trailing whitespaces and trim separating spaces
        line = reduce(line);

        // find separator position
        size_t const separatorPosition = line.find_first_of(" ");
        string key;
        pair<string, size_t> value;

        if (separatorPosition == string::npos)
        {
            // first check for single keyword without value
            key = line;
            value = pair<string, size_t>("", i);
        }
        else
        {
            // one or more arguments
            key = line.substr(0, separatorPosition);
            value = pair<string, size_t>(line.erase(0, separatorPosition + 1),
                                         i);
        }

        contents.insert(pair<string, pair<string, size_t> >(key, value));
    }

    size_t numProblems = sanityCheck();
    if (numProblems > 0)
    {
        log.push_back(strpr("WARNING: %d problems detected.\n", numProblems));
    }

    log.push_back(strpr("Found %d lines with keywords.\n", contents.size()));

    return;
}

size_t Settings::sanityCheck()
{
    size_t countProblems = 0;

    // check for unknown keywords
    for (multimap<string, pair<string, size_t> >::const_iterator
         it = contents.begin(); it != contents.end(); ++it)
    {
        if (knownKeywords.find((*it).first) == knownKeywords.end())
        {
            countProblems++;
            log.push_back(strpr(
                "WARNING: Unknown keyword \"%s\".\n", (*it).first.c_str()));
        }
    }

    // check for multiple instances of known keywords (with exceptions)
    for (multimap<string, string>::const_iterator it = knownKeywords.begin();
         it != knownKeywords.end(); ++it)
    {
        if (contents.count((*it).first) > 1
            && (*it).first != "symfunction_short"
            && (*it).first != "atom_energy")
        {
            countProblems++;
            log.push_back(strpr(
                "WARNING: Multiple instances of \"%s\" detected.\n",
                (*it).first.c_str()));
        }
    }

    return countProblems;
}
