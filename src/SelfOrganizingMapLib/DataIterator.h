/**
 * @file   SelfOrganizingMapLib/DataIterator.h
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <algorithm>
#include <istream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "Data.h"
#include "UtilitiesLib/get_file_header.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// Lazy iterator for reading data
template <typename Layout, typename T>
class DataIterator
{
public:

    typedef Data<Layout, T> DataType;
    typedef std::shared_ptr<DataType> PtrDataType;

    /// Default constructor
    DataIterator(std::istream& is, bool end_flag)
     : number_of_entries(0),
       is(is),
       header_offset(0),
       count(0),
       end_flag(end_flag)
    {}

    /// Parameter constructor
    explicit DataIterator(std::istream& is)
     : number_of_entries(0),
       is(is),
       header_offset(0),
       count(0),
       end_flag(false)
    {
        // Skip header
        get_file_header(is);

        // Ignore first three entries
        is.seekg(3 * sizeof(int), is.cur);
        is.read(reinterpret_cast<char*>(&number_of_entries), sizeof(int));
        // Ignore layout and dimensionality
        is.seekg(2 * sizeof(int), is.cur);

        for (uint8_t i = 0; i < layout.dimensionality; ++i) {
            is.read(reinterpret_cast<char*>(&layout.m_dimension[i]), sizeof(int));
        }

        header_offset = is.tellg();

        next();
    }

    /// Equal comparison
    bool operator == (DataIterator const& other) const
    {
        return end_flag == other.end_flag;
    }

    /// Unequal comparison
    bool operator != (DataIterator const& other) const
    {
        return !operator==(other);
    }

    /// Prefix increment
    DataIterator& operator ++ ()
    {
        next();
        return *this;
    }

    /// Addition assignment operator
    DataIterator& operator += (int steps)
    {
        is.seekg((steps - 1) * layout.size() * sizeof(T), is.cur);
        count += steps - 1;
        next();
        return *this;
    }

    /// Dereference
    DataType& operator * () const
    {
        return *ptr_current_entry;
    }

    /// Dereference
    DataType* operator -> () const
    {
        return &(operator*());
    }

private:

    /// Read next entry
    void next()
    {
        if (count < number_of_entries) {
            ptr_current_entry = std::make_shared<DataType>(layout);
            is.read(reinterpret_cast<char*>(ptr_current_entry->get_data_pointer()),
                static_cast<std::streamsize>(layout.size() * sizeof(T)));
            ++count;
        } else {
            is.seekg(0, is.beg);
            end_flag = true;
        }
    }

    uint32_t number_of_entries;

    std::istream& is;

    PtrDataType ptr_current_entry;

    std::streamoff header_offset;

    Layout layout;

    uint32_t count;

    /// Define the end iterator
    bool end_flag;
};

} // namespace pink
