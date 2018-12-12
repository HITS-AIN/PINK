/**
 * @file   SelfOrganizingMapLib/DataIterator.h
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Data.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// Lazy iterator with random access for reading data
template <typename Layout, typename T>
class DataIterator;

/// Lazy iterator with random access for reading data
template <typename T>
class DataIterator<CartesianLayout<2>, T>
{
public:

    typedef Data<CartesianLayout<2>, T> DataType;
    typedef std::shared_ptr<DataType> PtrDataType;

    /// Default constructor
    DataIterator()
     : number_of_entries(0),
	   ptr_stream(nullptr)
    {}

    /// Parameter constructor
    DataIterator(std::string const& filename)
     : number_of_entries(0),
	   ptr_stream(new std::ifstream(filename))
    {
        if (!(*ptr_stream)) throw pink::exception("DataIterator: Error opening " + filename);

        // Skip header lines
        std::string line;
        int last_position = ptr_stream->tellg();
        while (std::getline(*ptr_stream, line)) {
            if (line[0] != '#') break;
            last_position = ptr_stream->tellg();
        }

        ptr_stream->seekg(last_position, ptr_stream->beg);
        ptr_stream->read((char*)&number_of_entries, sizeof(int));
        ptr_stream->read((char*)&numberOfChannels_, sizeof(int));
        ptr_stream->read((char*)&height_, sizeof(int));
        ptr_stream->read((char*)&width_, sizeof(int));

        random_list.resize(number_of_entries);
        std::iota(std::begin(random_list), std::end(random_list), 0);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(std::begin(random_list), std::end(random_list), g);

        next();
    }

    /// Equal comparison
    bool operator == (DataIterator const& other) const
    {
        return ptrStream_ == other.ptrStream_;
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
    DataIterator& operator += (int step)
    {
        ptrStream_->seekg((step - 1) * height_ * width_ * sizeof(float), ptrStream_->cur);
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
        	ptr_current_entry = std::make_shared<DataType>(height_, width_, numberOfChannels_);
        	ptr_stream->read((char*)&ptr_current_entry->get_data_pointer(), numberOfChannels_ * height_ * width_ * sizeof(T));
            ++cur_random_list;
        } else {
        	ptr_stream.reset();
        }
    }

    uint32_t number_of_entries;

    std::vector<uint32_t> random_list;

    std::vector<uint32_t>::const_iterator cur_random_list;

    std::shared_ptr<std::ifstream> ptr_stream;

    PtrDataType ptr_current_entry;
};

} // namespace pink
