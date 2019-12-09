/**
 * @file   SelfOrganizingMapLib/DataIteratorShuffled.h
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <algorithm>
#include <istream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Data.h"
#include "UtilitiesLib/get_file_header.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// Lazy iterator with random access for reading data
template <typename Layout, typename T>
class DataIteratorShuffled
{
public:

    typedef Data<Layout, T> DataType;
    typedef std::shared_ptr<DataType> PtrDataType;

    /// Default constructor
    DataIteratorShuffled(std::istream& is, bool end_flag)
     : number_of_entries(0),
       is(is),
       header_offset(0),
       end_flag(end_flag)
    {}

    /// Parameter constructor
    DataIteratorShuffled(std::istream& is, uint64_t seed, bool do_shuffle = true)
     : number_of_entries(0),
       is(is),
       header_offset(0),
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

        random_list.resize(number_of_entries);
        std::iota(std::begin(random_list), std::end(random_list), 0);

        if (do_shuffle) {
            std::default_random_engine engine(seed);
            std::mt19937 dist(engine());
            std::shuffle(std::begin(random_list), std::end(random_list), dist);
        }

        cur_random_list = std::begin(random_list);

        next();
    }

    /// Equal comparison
    bool operator == (DataIteratorShuffled const& other) const
    {
        return end_flag == other.end_flag;
    }

    /// Unequal comparison
    bool operator != (DataIteratorShuffled const& other) const
    {
        return !operator==(other);
    }

    /// Prefix increment
    DataIteratorShuffled& operator ++ ()
    {
        next();
        return *this;
    }

    /// Addition assignment operator
    DataIteratorShuffled& operator += (int steps)
    {
        cur_random_list += steps;
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
        if (cur_random_list != std::end(random_list)) {
            is.seekg(header_offset + static_cast<std::streamoff>(
                *cur_random_list * layout.size() * sizeof(T)), is.beg);
            ptr_current_entry = std::make_shared<DataType>(layout);
            is.read(reinterpret_cast<char*>(ptr_current_entry->get_data_pointer()),
                static_cast<std::streamsize>(layout.size() * sizeof(T)));
            ++cur_random_list;
        } else {
            is.seekg(0, is.beg);
            end_flag = true;
        }
    }

    uint32_t number_of_entries;

    std::vector<uint32_t> random_list;

    std::vector<uint32_t>::const_iterator cur_random_list;

    std::istream& is;

    PtrDataType ptr_current_entry;

    std::streamoff header_offset;

    Layout layout;

    /// Define the end iterator
    bool end_flag;
};

} // namespace pink
