#pragma once

#include <stdexcept>
#include <string>

namespace pink {

struct exception : public std::runtime_error
{
    explicit exception(std::string const& msg)
     : std::runtime_error(msg)
    {}
};

} // namespace pink
