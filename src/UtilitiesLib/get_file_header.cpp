/**
 * @file   UtilitiesLib/get_file_header.cpp
 * @date   Jun 11, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "get_file_header.h"

namespace pink {

std::string get_file_header(std::istream& is)
{
    std::string header, line;
    auto last_position{is.tellg()};

    while (std::getline(is, line) && line[0] == '#')
    {
        last_position = is.tellg();
        if (line == "# END OF HEADER") {
            header += line + '\n';
            break;
        } else {
            header += line + '\n';
        }
    }

    is.clear();
    is.seekg(last_position, is.beg);

    return header;
}

} // namespace pink
