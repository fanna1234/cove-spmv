#pragma once

#include <cstdint>

namespace structural {

using Index = std::int32_t;
using Offset = std::int32_t;
using BitmapWord = std::uint32_t;

constexpr int kBitmapWordBits = static_cast<int>(sizeof(BitmapWord) * 8);

}  // namespace structural
