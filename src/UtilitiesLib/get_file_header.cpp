/**
 * @file   UtilitiesLib/get_file_header.cpp
 * @date   Jun 11, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "get_file_header.h"

namespace pink {

std::string get_file_header(std::istream& ifs)
{
    std::string header, line;
    auto last_position{ifs.tellg()};

    while (std::getline(ifs, line) && line[0] == '#')
    {
        if (line == "# END OF HEADER") {
            header = line + '\n';
            break;
        } else {
            header = line + '\n';
            last_position = ifs.tellg();
        }
    }

    ifs.clear();
    ifs.seekg(last_position, ifs.beg);

    return header;
}

} // namespace pink
