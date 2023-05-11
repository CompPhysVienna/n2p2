//
// Created by philipp on 2/21/23.
//

#ifndef N2P2_KEY_H
#define N2P2_KEY_H

#include <string>
#include <vector>
#include <iterator>

namespace nnp
{
    namespace settings
    {
        /// Keyword properties.
        class Key
        {
        public:
            std::string getMainKeyword() const
            {
                return keywords.at(0);
            }
            /// Whether this keyword has no alternative definitions or spellings.
            bool hasUniqueKeyword() const {return (keywords.size() == 1);}
            void addAlternative(std::string word) {keywords.push_back(word);};
            void setDescription(std::string text) {description = text;};

            std::vector<std::string>::const_iterator begin() const
            {
                return keywords.begin();
            }
            std::vector<std::string>::const_iterator end() const
            {
                return keywords.end();
            }

        private:
            /// A short description of the keyword.
            std::string              description;
            /// Alternative keywords (first entry is main name).
            std::vector<std::string> keywords;
        };
    }
}
#endif //N2P2_KEY_H
