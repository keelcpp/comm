#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::comm {

enum class CommStatus : u8 {
    kOk = 0,
    kInvalidArgument,
    kNotSystematic,
    kDecodeFailed,
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

enum class CyclPolyMode : u8 {
    kFirst = 0,
    kMin,
    kMax,
    kAll,
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

// M-FSK baseband modulation & non-coherent matched-filter demodulation.
// `fsk_mod` writes `symbols.size() * nsamp` complex samples.
// `fsk_demod` writes `samples.size() / nsamp` integer symbols.
[[nodiscard]] CommStatus fsk_mod(ArrayRef<i32> symbols, i32 order, f64 freq_sep, i32 nsamp, f64 fs,
                                 bool continuous_phase, SymbolOrder mapping,
                                 MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus fsk_demod(ArrayRef<c64> samples, i32 order, f64 freq_sep, i32 nsamp,
                                   f64 fs, SymbolOrder mapping, MutArrayRef<i32> out,
                                   Allocator* alloc = nullptr);

// OFDM modulation & demodulation.
// `in_col_major` is `nfft x nsym` (column-major); `out` has `(nfft + cplen) * nsym` samples.
[[nodiscard]] CommStatus ofdm_mod(ArrayRef<c64> in_col_major, usize nfft, usize nsym, usize cplen,
                                  MutArrayRef<c64> out, Allocator* alloc = nullptr);
// `in` has `(nfft + cplen) * nsym` samples; `out_col_major` is `nfft x nsym`.
[[nodiscard]] CommStatus ofdm_demod(ArrayRef<c64> in, usize nfft, usize nsym, usize cplen,
                                    usize sym_offset, MutArrayRef<c64> out_col_major,
                                    Allocator* alloc = nullptr);

// Multi-ring APSK constellation & generic constellation mod/demod.
// `phase_offsets` may be empty (defaults to pi / ring_sizes[r]).
[[nodiscard]] CommStatus apsk_constellation(ArrayRef<usize> ring_sizes, ArrayRef<f64> radii,
                                            ArrayRef<f64> phase_offsets, Array<c64>& out,
                                            Allocator* alloc = nullptr);
[[nodiscard]] CommStatus genqam_mod(ArrayRef<i32> symbols, ArrayRef<c64> constellation,
                                    MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus genqam_demod(ArrayRef<c64> samples, ArrayRef<c64> constellation,
                                      MutArrayRef<i32> out) noexcept;

// Analog & MSK modulation / demodulation (1-D channel kernels).
[[nodiscard]] CommStatus pm_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 phase_dev, f64 init_phase,
                                MutArrayRef<f64> out) noexcept;
[[nodiscard]] CommStatus pm_demod(ArrayRef<f64> y, f64 fc, f64 fs, f64 phase_dev, f64 init_phase,
                                  MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] CommStatus am_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 init_phase, f64 carrier_amp,
                                MutArrayRef<f64> out) noexcept;
[[nodiscard]] CommStatus fm_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 freq_dev, f64 init_phase,
                                MutArrayRef<f64> out) noexcept;
[[nodiscard]] CommStatus fm_demod(ArrayRef<f64> y, f64 fc, f64 fs, f64 freq_dev, f64 init_phase,
                                  MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] CommStatus ssb_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 init_phase, bool upper,
                                 MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] CommStatus msk_mod_diff(ArrayRef<u8> bits, i32 nsamp, f64 init_phase,
                                      MutArrayRef<c64> out, Allocator* alloc = nullptr);
[[nodiscard]] CommStatus msk_demod_diff(ArrayRef<c64> samples, i32 nsamp, f64 init_phase,
                                        MutArrayRef<u8> out_bits, f64* phase_out = nullptr) noexcept;

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

// ============================================================================
// Linear Block Coding (GF(2))
// ============================================================================

struct BlockCodeMatrices {
    Array<u8> h; // (n - k) x n column-major parity-check matrix (0/1)
    Array<u8> g; // k x n column-major generator matrix (0/1)
    usize n = 0;
    usize k = 0;
};

struct CyclPolyResult {
    Array<u8> rows_col_major; // num_polys x (n - k + 1) column-major (0/1)
    usize num_polys = 0;
    usize poly_len = 0;
};

// Default GF(2) primitive polynomial bitmask for degree m in [2, 16] (0 if out of range).
[[nodiscard]] u64 default_prim_poly(i32 m) noexcept;

// Systematic generator <-> parity-check matrix conversion (column-major, 0/1).
// Input is `rows x cols` (rows < cols); output is `(cols - rows) x cols` column-major.
[[nodiscard]] CommStatus gen2par(ArrayRef<u8> mat_col_major, usize rows, usize cols,
                                 Array<u8>& out_col_major, Allocator* alloc = nullptr);

// Hamming code generator/parity matrices for order `m >= 2`.
// `prim_mask == 0` selects `default_prim_poly(m)`.
[[nodiscard]] CommStatus hammgen(i32 m, u64 prim_mask, BlockCodeMatrices& out,
                                 Allocator* alloc = nullptr);

// Generator polynomials over GF(2) of degree `n - k` dividing `x^n + 1` (ascending powers).
[[nodiscard]] CommStatus cyclpoly(i32 n, i32 k, CyclPolyMode mode, CyclPolyResult& out,
                                  Allocator* alloc = nullptr);

// Cyclic code generator & parity-check matrices from ascending-power `gen_poly`.
[[nodiscard]] CommStatus cyclgen(i32 n, ArrayRef<u8> gen_poly, bool systematic,
                                 BlockCodeMatrices& out, Allocator* alloc = nullptr);

// Standard-array syndrome decoding table (`2^nk x n` column-major, 0/1) for `nk x n` matrix `h`.
[[nodiscard]] CommStatus syndtable(ArrayRef<u8> h_col_major, usize nk, usize n,
                                   MutArrayRef<u8> out_col_major, Allocator* alloc = nullptr);

// Linear block encoder: `msg_words` (`num_words x k` row-major) * `g` (`k x n` col-major).
[[nodiscard]] CommStatus block_encode(ArrayRef<u8> msg_words_row_major, usize num_words,
                                      ArrayRef<u8> g_col_major, usize k, usize n,
                                      MutArrayRef<u8> code_words_row_major) noexcept;

// Syndrome-based systematic linear block decoder.
// `code_words` is `num_words x n` row-major; writes `num_words x k` row-major `msg_words`
// and per-word corrected bit count into `err_counts` (`num_words`).
[[nodiscard]] CommStatus block_decode(ArrayRef<u8> code_words_row_major, usize num_words,
                                      ArrayRef<u8> g_col_major, ArrayRef<u8> h_col_major, usize k,
                                      usize n, MutArrayRef<u8> msg_words_row_major,
                                      MutArrayRef<i32> err_counts, Allocator* alloc = nullptr);

// ============================================================================
// Convolutional Coding & Viterbi Decoding
// ============================================================================

struct Trellis {
    usize num_input_symbols = 0;
    usize num_output_symbols = 0;
    usize num_states = 0;
    Array<u32> next_states; // num_states x num_input_symbols (column-major)
    Array<u32> outputs;     // num_states x num_input_symbols (column-major)
};

// Rate-1/n feed-forward convolutional trellis from constraint length K and octal code generators.
[[nodiscard]] CommStatus poly2trellis(i32 constraint_len, ArrayRef<u64> octal_generators,
                                      Trellis& out, Allocator* alloc = nullptr);

// Rate-1/n convolutional encoder (`out_bits` has length `msg_bits.size() * out_bits_per_sym`).
[[nodiscard]] CommStatus convenc(ArrayRef<u8> msg_bits, ArrayRef<u32> next_states_col_major,
                                 ArrayRef<u32> outputs_col_major, usize num_states,
                                 i32 out_bits_per_sym, MutArrayRef<u8> out_bits) noexcept;

// Rate-1/n hard-decision Viterbi decoder (`terminated == true` starts traceback at state 0).
[[nodiscard]] CommStatus vitdec_hard(ArrayRef<u8> code_bits, ArrayRef<u32> next_states_col_major,
                                     ArrayRef<u32> outputs_col_major, usize num_states,
                                     i32 out_bits_per_sym, bool terminated,
                                     MutArrayRef<u8> out_bits, Allocator* alloc = nullptr);

// ============================================================================
// Source Coding (Huffman, Arithmetic, Quantization, Lloyd-Max, DPCM)
// ============================================================================

struct HuffmanDict {
    Array<u32> offsets; // length K + 1; symbol i code is bits[offsets[i] .. offsets[i+1])
    Array<u8> bits;     // concatenated 0/1 code bits
    f64 avg_len = 0.0;
};

struct LloydsResult {
    Array<f64> partition; // length K - 1
    Array<f64> codebook;  // length K
    f64 distortion = 0.0;
    f64 rel_distortion = 0.0;
};

struct DpcmOptResult {
    Array<f64> predictor; // length order + 1, predictor[0] == 0
    Array<f64> partition; // empty if no quantizer requested
    Array<f64> codebook;  // empty if no quantizer requested
};

[[nodiscard]] CommStatus huffman_dict(ArrayRef<f64> probs, HuffmanDict& out,
                                      Allocator* alloc = nullptr);
[[nodiscard]] CommStatus huffman_encode(ArrayRef<usize> sym_indices, ArrayRef<u32> offsets,
                                        ArrayRef<u8> dict_bits, Array<u8>& out_bits,
                                        Allocator* alloc = nullptr);
[[nodiscard]] CommStatus huffman_decode(ArrayRef<u8> bits, ArrayRef<u32> offsets,
                                        ArrayRef<u8> dict_bits, Array<usize>& out_sym_indices,
                                        Allocator* alloc = nullptr);

// Arithmetic coding for 1-based symbol indices `seq_1based[i] in [1, counts.size()]`.
[[nodiscard]] CommStatus arith_encode(ArrayRef<usize> seq_1based, ArrayRef<u64> counts,
                                      Array<u8>& out_bits, Allocator* alloc = nullptr);
[[nodiscard]] CommStatus arith_decode(ArrayRef<u8> code_bits, ArrayRef<u64> counts, usize len,
                                      MutArrayRef<usize> out_seq_1based,
                                      Allocator* alloc = nullptr);

// Scalar quantizer: `out_idx[i] = count(partition < sig[i]) in [0, partition.size()]`.
[[nodiscard]] CommStatus quantiz_index(ArrayRef<f64> sig, ArrayRef<f64> partition,
                                       MutArrayRef<usize> out_idx) noexcept;
[[nodiscard]] CommStatus quantiz(ArrayRef<f64> sig, ArrayRef<f64> partition, ArrayRef<f64> codebook,
                                 MutArrayRef<usize> out_idx, MutArrayRef<f64> out_quant,
                                 f64& out_distor) noexcept;

// Lloyd-Max scalar quantizer designer (`init_codebook` sorted internally; or `init_k > 0`
// for uniform initial codebook across [min(training), max(training)]).
[[nodiscard]] CommStatus lloyds(ArrayRef<f64> training, ArrayRef<f64> init_codebook, usize init_k,
                                f64 tol, LloydsResult& out, Allocator* alloc = nullptr);

// Differential Pulse Code Modulation (`predictor` has length M + 1 with leading placeholder).
[[nodiscard]] CommStatus dpcm_encode(ArrayRef<f64> sig, ArrayRef<f64> codebook,
                                     ArrayRef<f64> partition, ArrayRef<f64> predictor,
                                     MutArrayRef<usize> out_idx, MutArrayRef<f64> out_qerr,
                                     Allocator* alloc = nullptr);
[[nodiscard]] CommStatus dpcm_decode(ArrayRef<usize> idx, ArrayRef<f64> codebook,
                                     ArrayRef<f64> predictor, MutArrayRef<f64> out_sig,
                                     MutArrayRef<f64> out_qerr, Allocator* alloc = nullptr);

// Yule-Walker + Levinson-Durbin DPCM predictor (and optional Lloyd-Max residual quantizer).
// Pass `init_codebook` empty and `init_k == 0` to skip residual quantizer design.
[[nodiscard]] CommStatus dpcm_opt(ArrayRef<f64> training, i32 order, ArrayRef<f64> init_codebook,
                                  usize init_k, DpcmOptResult& out, Allocator* alloc = nullptr);

// ============================================================================
// MIL-STD-188-110 QAM & Channel Metrics (qfunc, berawgn, noisebw)
// ============================================================================

// Returns MIL-STD-188-110 QAM constellation table for order in {16, 32, 64, 256} (empty otherwise).
[[nodiscard]] ArrayRef<c64> mil188_constellation(i32 order) noexcept;

[[nodiscard]] CommStatus mil188_qam_mod(ArrayRef<i32> symbols, i32 order,
                                        MutArrayRef<c64> out) noexcept;
[[nodiscard]] CommStatus mil188_qam_demod(ArrayRef<c64> samples, i32 order,
                                          MutArrayRef<i32> out) noexcept;

// Standard Gaussian Q-function Q(x) = 0.5 * erfc(x / sqrt(2)).
[[nodiscard]] f64 qfunc(f64 x) noexcept;

enum class AwgnMod : u8 {
    kPsk = 0,
    kQam,
    kPam,
    kFsk,
    kDpsk,
};

// Analytical BER in AWGN for `ebno_db` (dB), modulation family `mod`, and constellation size `order`.
[[nodiscard]] f64 ber_awgn(f64 ebno_db, AwgnMod mod, i32 order) noexcept;

// Equivalent noise bandwidth of discrete filter H(z) = B(z) / A(z) sampled over `n_samp` points.
[[nodiscard]] f64 noise_bw(ArrayRef<f64> num, ArrayRef<f64> den, i32 n_samp, f64 fs) noexcept;

} // namespace keel::comm
