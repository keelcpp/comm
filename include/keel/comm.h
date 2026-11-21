#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/ref.h"

namespace keel::comm {

enum class CommStatus : u8 {
    kOk = 0,
    kInvalidArgument,
};

enum class SymbolOrder : u8 {
    kGray = 0,
    kBinary = 1,
};

enum class NormMode : u8 {
    kAveragePower = 0,
    kPeakPower = 1,
};

enum class RcosShape : u8 {
    kNormal = 0,
    kSqrt = 1,
};

enum class CompandMode : u8 {
    kMuCompress = 0,
    kMuExpand = 1,
    kACompress = 2,
    kAExpand = 3,
};

struct ErrorStats {
    u64 count = 0;
    f64 ratio = 0.0;
};

// ============================================================================
// Gray Code Utilities
// ============================================================================

[[nodiscard]] constexpr u32 to_gray(u32 b) noexcept {
    return b ^ (b >> 1u);
}

[[nodiscard]] constexpr u32 from_gray(u32 g) noexcept {
    u32 b = g;
    while ((g >>= 1u) != 0u) {
        b ^= g;
    }
    return b;
}

// ============================================================================
// Digital Modulation & Demodulation
// ============================================================================

[[nodiscard]] CommStatus psk_mod(ArrayRef<i32> symbols, i32 order, f64 init_phase,
                                 SymbolOrder mapping, MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus psk_demod(ArrayRef<c64> samples, i32 order, f64 init_phase,
                                   SymbolOrder mapping, MutArrayRef<i32> out) noexcept;

[[nodiscard]] CommStatus dpsk_mod(ArrayRef<i32> symbols, i32 order, f64 phase_rot,
                                  SymbolOrder mapping, MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus dpsk_demod(ArrayRef<c64> samples, i32 order, f64 phase_rot,
                                    SymbolOrder mapping, MutArrayRef<i32> out) noexcept;

[[nodiscard]] CommStatus pam_mod(ArrayRef<i32> symbols, i32 order, f64 init_phase,
                                 SymbolOrder mapping, MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus pam_demod(ArrayRef<c64> samples, i32 order, f64 init_phase,
                                   SymbolOrder mapping, MutArrayRef<i32> out) noexcept;

[[nodiscard]] CommStatus qam_mod(ArrayRef<i32> symbols, i32 order, SymbolOrder mapping,
                                 bool unit_power, MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus qam_demod(ArrayRef<c64> samples, i32 order, SymbolOrder mapping,
                                   bool unit_power, MutArrayRef<i32> out) noexcept;

// Normalization scale factor so constellation achieves `target_power`.
[[nodiscard]] f64 modnorm(ArrayRef<c64> ref_constellation, NormMode mode,
                          f64 target_power = 1.0) noexcept;

// ============================================================================
// Pulse-Shaping FIR Design & Companding
// ============================================================================

// Raised-cosine / root-raised-cosine FIR (unit energy).
// Writes `span * sps + 1` taps into `out`.
[[nodiscard]] CommStatus rcos_design(f64 beta, usize span, usize sps, RcosShape shape,
                                     MutArrayRef<f64> out) noexcept;

// Gaussian pulse-shaping FIR (unit area, GMSK).
// Writes `span * sps + 1` taps into `out`.
[[nodiscard]] CommStatus gauss_design(f64 bt, usize span, usize sps, MutArrayRef<f64> out) noexcept;

// Mu-law and A-law compressor / expander (`out.size() >= in.size()`).
[[nodiscard]] CommStatus compand(ArrayRef<f64> in, f64 param, f64 peak_v, CompandMode mode,
                                 MutArrayRef<f64> out) noexcept;

// ============================================================================
// Multiplicative Scrambling & Error Metrics
// ============================================================================

// Self-synchronizing multiplicative scrambler / descrambler.
// `poly` has length `degree + 1` with `poly[0] != 0`; `state` has length `degree`
// and is updated in-place; `out_bits.size() >= in_bits.size()`.
[[nodiscard]] CommStatus scramble(ArrayRef<u8> in_bits, ArrayRef<u8> poly, MutArrayRef<u8> state,
                                  MutArrayRef<u8> out_bits) noexcept;
[[nodiscard]] CommStatus descramble(ArrayRef<u8> in_bits, ArrayRef<u8> poly, MutArrayRef<u8> state,
                                    MutArrayRef<u8> out_bits) noexcept;

// Bit error count and ratio between non-negative integer symbol streams `a` and `b`.
// If `bits_per_symbol <= 0`, automatically selects minimal bit width covering max(a, b).
[[nodiscard]] ErrorStats bit_error_rate(ArrayRef<u64> a, ArrayRef<u64> b,
                                        i32 bits_per_symbol = 0) noexcept;

// Symbol error count and ratio between `a` and `b`.
[[nodiscard]] ErrorStats sym_error_rate(ArrayRef<f64> a, ArrayRef<f64> b) noexcept;

} // namespace keel::comm
