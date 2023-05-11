//
// Created by philipp on 2/21/23.
//

#ifndef N2P2_ISETTINGS_H
#define N2P2_ISETTINGS_H

#include "Key.h"

namespace nnp
{
    namespace settings
    {
        class ISettings
        {
        public:
            virtual bool keywordExists(Key const& key,
                                       bool const exact = false) const = 0;
            virtual std::string getValue(Key const& key) const = 0;
        };
    }
}

#endif //N2P2_ISETTINGS_H
