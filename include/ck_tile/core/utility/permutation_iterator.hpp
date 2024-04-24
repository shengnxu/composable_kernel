// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck_tile/core/utility/iterator.hpp"
#include "ck_tile/core/utility/iterator_range.hpp"

namespace ck_tile {

template <typename ElementIterator, typename IndexIterator>
struct permutation_iterator
{
    static_assert(is_random_access_iterator_v<ElementIterator>);

    using element_iterator  = ElementIterator;
    using index_iterator    = IndexIterator;
    using reference         = iter_reference_t<element_iterator>;
    using difference_type   = iter_difference_t<index_iterator>;
    using value_type        = iter_value_t<element_iterator>;
    using pointer           = typename std::iterator_traits<element_iterator>::pointer;
    using iterator_category = typename std::iterator_traits<index_iterator>::iterator_category;

    permutation_iterator() = delete;

    explicit constexpr permutation_iterator(element_iterator next_element_,
                                            index_iterator next_index_)
        : next_element(next_element_), next_index(next_index_)
    {
    }

    reference operator*() const { return *std::next(base(), *index()); }

    permutation_iterator& operator++()
    {
        ++next_index;
        return *this;
    }

    permutation_iterator operator++(int)
    {
        permutation_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    permutation_iterator& operator--()
    {
        --next_index;
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    permutation_iterator operator--(int)
    {
        permutation_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, permutation_iterator&> operator+=(difference_type step)
    {
        std::advance(next_index, step);
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, permutation_iterator> operator-=(difference_type step)
    {
        return (*this) += (-step);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, reference> operator[](difference_type step) const
    {
        return *std::next(base(), *std::next(index(), step));
    }

    element_iterator base() const { return next_element; }

    private:
    index_iterator index() const { return next_index; }

    friend bool operator==(const permutation_iterator& lhs, const permutation_iterator& rhs)
    {
        return lhs.base() == rhs.base() && lhs.index() == rhs.index();
    }

    friend bool operator!=(const permutation_iterator& lhs, const permutation_iterator& rhs)
    {
        return !(lhs == rhs);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, permutation_iterator> operator+(const permutation_iterator& lhs,
                                                                  difference_type step)
    {
        permutation_iterator result(lhs);
        result += step;
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, permutation_iterator> operator-(const permutation_iterator& lhs,
                                                                  difference_type step)
    {
        return lhs + (-step);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend difference_type operator-(const permutation_iterator& lhs,
                                     const permutation_iterator& rhs)
    {
        return lhs.index() - rhs.index();
    }

    element_iterator next_element;
    index_iterator next_index;
};

template <typename ElementIterator, typename IndexIterator>
permutation_iterator(ElementIterator, IndexIterator)
    -> permutation_iterator<ElementIterator, IndexIterator>;

template <typename ElementRange, typename IndexRange>
constexpr auto make_permutation_range(ElementRange&& elements, IndexRange&& indices)
{
    using std::begin, std::end;

    return iterator_range(permutation_iterator(begin(elements), begin(indices)),
                          permutation_iterator(begin(elements), end(indices)));
}

} // namespace ck_tile