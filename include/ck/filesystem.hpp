// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef GUARD_CK_FILESYSTEM_HPP_
#define GUARD_CK_FILESYSTEM_HPP_

#include <string>
#include <string_view>

// clang-format off
#if defined(CPPCHECK)
  #define CK_HAS_FILESYSTEM 1
  #define CK_HAS_FILESYSTEM_TS 1
#elif defined(_WIN32)
  #if _MSC_VER >= 1920
    #define CK_HAS_FILESYSTEM 1
    #define CK_HAS_FILESYSTEM_TS 0
  #elif _MSC_VER >= 1900
    #define CK_HAS_FILESYSTEM 0
    #define CK_HAS_FILESYSTEM_TS 1
  #else
    #define CK_HAS_FILESYSTEM 0
    #define CK_HAS_FILESYSTEM_TS 0
  #endif
#elif defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L
    #define CK_HAS_FILESYSTEM 1
  #else
    #define CK_HAS_FILESYSTEM 0
  #endif
  #if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
    #define CK_HAS_FILESYSTEM_TS 1
  #else
    #define CK_HAS_FILESYSTEM_TS 0
  #endif
#else
  #define CK_HAS_FILESYSTEM 0
  #define CK_HAS_FILESYSTEM_TS 0
#endif
// clang-format on

#if CK_WORKAROUND_USE_BOOST_FILESYSTEM
#undef CK_HAS_FILESYSTEM
#undef CK_HAS_FILESYSTEM_TS
#define CK_HAS_FILESYSTEM 0
#define CK_HAS_FILESYSTEM_TS 0
#endif

#if CK_HAS_FILESYSTEM
#include <filesystem>
#elif CK_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#elif CK_WORKAROUND_USE_BOOST_FILESYSTEM
/// Explicit inclusion of <boost/container_hash/hash.hpp> is a workaround for the Boost 1.83 issue.
///
/// Boost doc is saying:
///   When writing template classes, you might not want to include the main hash.hpp header as it's
///   quite an expensive include that brings in a lot of other headers, so instead you can include
///   the <boost/container_hash/hash_fwd.hpp> header which forward declares boost::hash,
///   boost::hash_combine, boost::hash_range, and boost::hash_unordered_range. You'll need to
///   include the main header before instantiating boost::hash.
///
/// \ref https://www.boost.org/doc/libs/1_83_0/libs/container_hash/doc/html/hash.html#combine
///
/// It seems like boost::filesystem uses boost::hash_range but misses to include hash.hpp, so we
/// have to include it explicitly. Otherwise an error like this happens at runtime (!):
///
/// \code
/// ...symbol lookup error: .../libCK.so.1: undefined symbol:
/// _ZN5boost10hash_rangeIN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEEmT_SC_
/// \endcode
#include <boost/container_hash/hash.hpp>
#define BOOST_FILESYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>
#else
#error "No filesystem include available"
#endif

namespace CK {

#if CK_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif CK_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#elif CK_WORKAROUND_USE_BOOST_FILESYSTEM
namespace fs = ::boost::filesystem;
#endif

} // namespace CK

inline std::string operator+(const std::string_view s, const CK::fs::path& path)
{
#if CK_WORKAROUND_USE_BOOST_FILESYSTEM
    return std::string(path.native()).insert(0, s);
#else
    return path.string().insert(0, s);
#endif
}

inline std::string operator+(const CK::fs::path& path, const std::string_view s)
{
#if CK_WORKAROUND_USE_BOOST_FILESYSTEM
    return std::string(path.native()).append(s);
#else
    return path.string().append(s);
#endif
}

// Hack to minimize changes for boost fs.
#if CK_WORKAROUND_USE_BOOST_FILESYSTEM
#define FS_ENUM_PERMS_ALL fs::perms::all_all
#else
#define FS_ENUM_PERMS_ALL fs::perms::all
#endif

#if CK_HAS_FILESYSTEM_TS
#ifdef __linux__
#include <linux/limits.h>
namespace CK {
inline fs::path weakly_canonical(const fs::path& path)
{
    std::string result(PATH_MAX, '\0');
    std::string p{path.is_relative() ? (fs::current_path() / path).string() : path.string()};
    char* retval = realpath(p.c_str(), &result[0]);
    return (retval == nullptr) ? path : fs::path{result};
}
} // namespace CK
#else
#error "Not implmeneted!"
#endif
#else
namespace CK {
inline fs::path weakly_canonical(const fs::path& path) { return fs::weakly_canonical(path); }
} // namespace CK
#endif

namespace CK {

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view dynamic_library_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view dynamic_library_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

inline fs::path make_executable_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + executable_postfix);
}

inline fs::path make_dynamic_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + dynamic_library_postfix);
}

inline fs::path make_object_file_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + object_file_postfix);
}

inline fs::path make_static_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + static_library_postfix);
}

struct FsPathHash
{
    std::size_t operator()(const fs::path& path) const { return fs::hash_value(path); }
};
} // namespace CK

#endif // GUARD_CK_FILESYSTEM_HPP_