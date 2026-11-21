#include "keel/comm.h"

#include "keel/math.h"

namespace keel::comm {

namespace {

[[nodiscard]] u64 popcount64(u64 x) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    return as<u64>(__builtin_popcountll(x));
#else
    u64 c = 0;
    while (x != 0) {
        c += (x & 1ULL);
        x >>= 1u;
    }
    return c;
#endif
}

} // namespace

CommStatus scramble(ArrayRef<u8> in_bits, ArrayRef<u8> poly, MutArrayRef<u8> state,
                    MutArrayRef<u8> out_bits) noexcept {
    if (poly.size() < 2 || poly[0] == 0) {
        return CommStatus::kInvalidArgument;
    }
    const usize deg = poly.size() - 1;
    if (state.size() != deg || out_bits.size() < in_bits.size()) {
        return CommStatus::kInvalidArgument;
    }
    for (usize k = 0; k < in_bits.size(); ++k) {
        u8 fb = 0;
        for (usize i = 1; i <= deg; ++i) {
            if (poly[i] != 0) {
                fb ^= (state[i - 1] & 1u);
            }
        }
        const u8 yk = as<u8>((in_bits[k] & 1u) ^ fb);
        out_bits[k] = yk;
        for (usize i = deg; i-- > 1;) {
            state[i] = state[i - 1];
        }
        state[0] = yk;
    }
    return CommStatus::kOk;
}

CommStatus descramble(ArrayRef<u8> in_bits, ArrayRef<u8> poly, MutArrayRef<u8> state,
                      MutArrayRef<u8> out_bits) noexcept {
    if (poly.size() < 2 || poly[0] == 0) {
        return CommStatus::kInvalidArgument;
    }
    const usize deg = poly.size() - 1;
    if (state.size() != deg || out_bits.size() < in_bits.size()) {
        return CommStatus::kInvalidArgument;
    }
    for (usize k = 0; k < in_bits.size(); ++k) {
        const u8 yk = as<u8>(in_bits[k] & 1u);
        u8 fb = 0;
        for (usize i = 1; i <= deg; ++i) {
            if (poly[i] != 0) {
                fb ^= (state[i - 1] & 1u);
            }
        }
        out_bits[k] = as<u8>(yk ^ fb);
        for (usize i = deg; i-- > 1;) {
            state[i] = state[i - 1];
        }
        state[0] = yk;
    }
    return CommStatus::kOk;
}

ErrorStats bit_error_rate(ArrayRef<u64> a, ArrayRef<u64> b, i32 bits_per_symbol) noexcept {
    const usize n = keel::min(a.size(), b.size());
    if (n == 0) {
        return ErrorStats{0, 0.0};
    }
    i32 k_bits = bits_per_symbol;
    if (k_bits < 1) {
        u64 mx = 0;
        for (usize i = 0; i < n; ++i) {
            mx = keel::max(mx, keel::max(a[i], b[i]));
        }
        k_bits = 1;
        while (k_bits < 64 && (u64{1} << as<u32>(k_bits)) <= mx) {
            ++k_bits;
        }
    }
    u64 total = 0;
    for (usize i = 0; i < n; ++i) {
        total += popcount64(a[i] ^ b[i]);
    }
    const f64 total_bits = as<f64>(n) * as<f64>(k_bits);
    const f64 ratio = (total_bits > 0.0) ? (as<f64>(total) / total_bits) : 0.0;
    return ErrorStats{total, ratio};
}

ErrorStats sym_error_rate(ArrayRef<f64> a, ArrayRef<f64> b) noexcept {
    const usize n = keel::min(a.size(), b.size());
    if (n == 0) {
        return ErrorStats{0, 0.0};
    }
    u64 count = 0;
    for (usize i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++count;
        }
    }
    return ErrorStats{count, as<f64>(count) / as<f64>(n)};
}

} // namespace keel::comm
