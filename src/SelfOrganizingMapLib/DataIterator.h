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
#include <random>
#include <string>
#include <vector>

#include "Data.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// Lazy iterator with random access for reading data
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
       end_flag(end_flag)
    {}

    /// Parameter constructor
    DataIterator(std::istream& is)
     : number_of_entries(0),
       is(is),
       end_flag(false)
    {
        // Skip header lines
        std::string line;
        int last_position = is.tellg();
        while (std::getline(is, line)) {
            if (line[0] != '#') break;
            last_position = is.tellg();
        }

        // Ignore first three entries
        is.seekg(last_position + 3 * sizeof(int), is.beg);
        is.read((char*)&number_of_entries, sizeof(int));
        // Ignore layout and dimensionality
        is.seekg(2 * sizeof(int), is.cur);

        for (int i = 0; i < layout.dimensionality; ++i) {
            is.read((char*)&layout.dimension[i], sizeof(int));
        }

        header_offset = is.tellg();

        random_list.resize(number_of_entries);
        std::iota(std::begin(random_list), std::end(random_list), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(std::begin(random_list), std::end(random_list), g);

        cur_random_list = std::begin(random_list);

        next();
    }

    /// Equal comparison
    bool operator == (DataIterator const& other) const
    {
        return end_flag == other.end_flag;// && is == other.is;
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

    /// Return number of images.
    int get_number_of_entries() const { return number_of_entries; }

private:

    /// Read next entry
    void next()
    {
        if (cur_random_list != std::end(random_list)) {
            is.seekg(header_offset + *cur_random_list * layout.size() * sizeof(T), is.beg);
            ptr_current_entry = std::make_shared<DataType>(layout);
            is.read((char*)ptr_current_entry->get_data_pointer(), layout.size() * sizeof(T));
            ++cur_random_list;
        } else {
            end_flag = true;
        }
    }

    uint32_t number_of_entries;

    std::vector<uint32_t> random_list;

    std::vector<uint32_t>::const_iterator cur_random_list;

    std::istream& is;

    PtrDataType ptr_current_entry;

    int header_offset;

    Layout layout;

    /// Define the end iterator
    bool end_flag;
};

} // namespace pink
