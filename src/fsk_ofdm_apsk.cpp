#include "keel/comm.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"
#include "keel/signal.h"

namespace keel::comm {

namespace {

inline i32 fsk_sym_to_idx(i32 s, SymbolOrder mapping) noexcept {
    return (mapping == SymbolOrder::kBinary) ? s : as<i32>(to_gray(as<u32>(s)));
}

inline i32 fsk_idx_to_sym(i32 k, SymbolOrder mapping) noexcept {
    return (mapping == SymbolOrder::kBinary) ? k : as<i32>(from_gray(as<u32>(k)));
}

} // namespace

// ============================================================================
// M-FSK Modulation & Demodulation
// ============================================================================

CommStatus fsk_mod(ArrayRef<i32> symbols, i32 order, f64 freq_sep, i32 nsamp, f64 fs,
                   bool continuous_phase, SymbolOrder mapping, MutArrayRef<c64> out) noexcept {
    if (order < 2 || freq_sep <= 0.0 || nsamp <= 0 || fs <= 0.0) {
        return CommStatus::kInvalidArgument;
    }
    const usize ns = as<usize>(nsamp);
    const usize nout = symbols.size() * ns;
    if (out.size() < nout) return CommStatus::kInvalidArgument;
    if (nout == 0) return CommStatus::kOk;

    f64 phase = 0.0;
    for (usize s = 0; s < symbols.size(); ++s) {
        const i32 k = fsk_sym_to_idx(symbols[s], mapping);
        const f64 f = (as<f64>(k) - 0.5 * as<f64>(order - 1)) * freq_sep;
        const f64 dphi = kTau * f / fs;
        for (usize n = 0; n < ns; ++n) {
            if (continuous_phase) {
                out[s * ns + n] = c64(keel::cos(phase), keel::sin(phase));
                phase += dphi;
            } else {
                const f64 th = dphi * as<f64>(n);
                out[s * ns + n] = c64(keel::cos(th), keel::sin(th));
            }
        }
    }
    return CommStatus::kOk;
}

CommStatus fsk_demod(ArrayRef<c64> samples, i32 order, f64 freq_sep, i32 nsamp, f64 fs,
                     SymbolOrder mapping, MutArrayRef<i32> out, Allocator* alloc) {
    if (order < 2 || freq_sep <= 0.0 || nsamp <= 0 || fs <= 0.0) {
        return CommStatus::kInvalidArgument;
    }
    const usize ns = as<usize>(nsamp);
    const usize nsym = samples.size() / ns;
    if (out.size() < nsym) return CommStatus::kInvalidArgument;
    if (nsym == 0) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> ref(as<usize>(order) * ns, c64(0.0, 0.0), scratch.allocator());
    for (i32 k = 0; k < order; ++k) {
        const f64 f = (as<f64>(k) - 0.5 * as<f64>(order - 1)) * freq_sep;
        const f64 dphi = kTau * f / fs;
        for (usize n = 0; n < ns; ++n) {
            const f64 th = dphi * as<f64>(n);
            ref[as<usize>(k) * ns + n] = c64(keel::cos(th), keel::sin(th));
        }
    }

    for (usize s = 0; s < nsym; ++s) {
        i32 best_k = 0;
        f64 best_e = -1.0;
        for (i32 k = 0; k < order; ++k) {
            c64 acc(0.0, 0.0);
            for (usize n = 0; n < ns; ++n) {
                acc += samples[s * ns + n] * keel::conj(ref[as<usize>(k) * ns + n]);
            }
            const f64 e = keel::norm(acc);
            if (e > best_e) {
                best_e = e;
                best_k = k;
            }
        }
        out[s] = fsk_idx_to_sym(best_k, mapping);
    }
    return CommStatus::kOk;
}

// ============================================================================
// OFDM Modulation & Demodulation
// ============================================================================

CommStatus ofdm_mod(ArrayRef<c64> in_col_major, usize nfft, usize nsym, usize cplen,
                    MutArrayRef<c64> out, Allocator* alloc) {
    if (nfft == 0 || cplen > nfft || in_col_major.size() < nfft * nsym) {
        return CommStatus::kInvalidArgument;
    }
    const usize out_per_sym = nfft + cplen;
    if (out.size() < out_per_sym * nsym) return CommStatus::kInvalidArgument;
    if (nsym == 0) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> buf(nfft, c64(0.0, 0.0), scratch.allocator());

    for (usize s = 0; s < nsym; ++s) {
        for (usize i = 0; i < nfft; ++i) buf[i] = in_col_major[s * nfft + i];
        keel::fft::ifft_1d_inplace(buf.as_mut_ref());

        const usize dst_off = s * out_per_sym;
        for (usize i = 0; i < cplen; ++i) {
            out[dst_off + i] = buf[nfft - cplen + i];
        }
        for (usize i = 0; i < nfft; ++i) {
            out[dst_off + cplen + i] = buf[i];
        }
    }
    return CommStatus::kOk;
}

CommStatus ofdm_demod(ArrayRef<c64> in, usize nfft, usize nsym, usize cplen, usize sym_offset,
                      MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (nfft == 0 || sym_offset > cplen) {
        return CommStatus::kInvalidArgument;
    }
    const usize out_per_sym = nfft + cplen;
    if (in.size() < out_per_sym * nsym || out_col_major.size() < nfft * nsym) {
        return CommStatus::kInvalidArgument;
    }
    if (nsym == 0) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> buf(nfft, c64(0.0, 0.0), scratch.allocator());

    for (usize s = 0; s < nsym; ++s) {
        const usize src_off = s * out_per_sym + sym_offset;
        for (usize i = 0; i < nfft; ++i) buf[i] = in[src_off + i];
        keel::fft::fft_1d_inplace(buf.as_mut_ref());
        for (usize i = 0; i < nfft; ++i) out_col_major[s * nfft + i] = buf[i];
    }
    return CommStatus::kOk;
}

// ============================================================================
// Multi-Ring APSK & Generic Constellation Mod/Demod
// ============================================================================

CommStatus apsk_constellation(ArrayRef<usize> ring_sizes, ArrayRef<f64> radii,
                              ArrayRef<f64> phase_offsets, Array<c64>& out, Allocator* alloc) {
    const usize r_cnt = ring_sizes.size();
    if (r_cnt == 0 || radii.size() != r_cnt) return CommStatus::kInvalidArgument;
    if (!phase_offsets.empty() && phase_offsets.size() != 1u && phase_offsets.size() != r_cnt) {
        return CommStatus::kInvalidArgument;
    }

    usize total = 0u;
    for (usize r = 0; r < r_cnt; ++r) {
        if (ring_sizes[r] == 0u) return CommStatus::kInvalidArgument;
        total += ring_sizes[r];
    }

    out = Array<c64>(alloc);
    out.reserve(total);
    for (usize r = 0; r < r_cnt; ++r) {
        const usize mr = ring_sizes[r];
        const f64 rad = radii[r];
        f64 pho = kPi / as<f64>(mr);
        if (phase_offsets.size() == 1u) {
            pho = phase_offsets[0];
        } else if (phase_offsets.size() == r_cnt) {
            pho = phase_offsets[r];
        }
        for (usize k = 0; k < mr; ++k) {
            const f64 theta = pho + kTau * as<f64>(k) / as<f64>(mr);
            out.push_back(c64(rad * keel::cos(theta), rad * keel::sin(theta)));
        }
    }
    return CommStatus::kOk;
}

CommStatus genqam_mod(ArrayRef<i32> symbols, ArrayRef<c64> constellation,
                      MutArrayRef<c64> out) noexcept {
    if (constellation.empty() || out.size() < symbols.size()) {
        return CommStatus::kInvalidArgument;
    }
    const i32 m = as<i32>(constellation.size());
    for (usize i = 0; i < symbols.size(); ++i) {
        if (symbols[i] < 0 || symbols[i] >= m) return CommStatus::kInvalidArgument;
        out[i] = constellation[as<usize>(symbols[i])];
    }
    return CommStatus::kOk;
}

CommStatus genqam_demod(ArrayRef<c64> samples, ArrayRef<c64> constellation,
                        MutArrayRef<i32> out) noexcept {
    if (constellation.empty() || out.size() < samples.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize m = constellation.size();
    for (usize i = 0; i < samples.size(); ++i) {
        const c64 yi = samples[i];
        usize best_k = 0u;
        f64 best_d2 = Limits<f64>::infinity();
        for (usize k = 0; k < m; ++k) {
            const f64 dr = yi.real() - constellation[k].real();
            const f64 di = yi.imag() - constellation[k].imag();
            const f64 d2 = dr * dr + di * di;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = k;
            }
        }
        out[i] = as<i32>(best_k);
    }
    return CommStatus::kOk;
}

// ============================================================================
// Analog & MSK Modulation / Demodulation
// ============================================================================

CommStatus pm_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 phase_dev, f64 init_phase,
                  MutArrayRef<f64> out) noexcept {
    if (!(fs > 0.0) || !(fc > 0.0) || fs < 2.0 * fc || !(phase_dev > 0.0) ||
        out.size() < x.size()) {
        return CommStatus::kInvalidArgument;
    }
    const f64 two_pi_fc = kTau * fc;
    const f64 inv_fs = 1.0 / fs;
    for (usize r = 0; r < x.size(); ++r) {
        const f64 t = as<f64>(r) * inv_fs;
        out[r] = keel::cos(two_pi_fc * t + phase_dev * x[r] + init_phase);
    }
    return CommStatus::kOk;
}

CommStatus pm_demod(ArrayRef<f64> y, f64 fc, f64 fs, f64 phase_dev, f64 init_phase,
                    MutArrayRef<f64> out, Allocator* alloc) {
    if (!(fs > 0.0) || !(fc > 0.0) || fs < 2.0 * fc || !(phase_dev > 0.0) ||
        out.size() < y.size()) {
        return CommStatus::kInvalidArgument;
    }
    if (y.empty()) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> analytic(y.size(), c64(0.0, 0.0), scratch.allocator());
    if (keel::signal::hilbert(y, analytic.as_mut_ref(), scratch.allocator()) !=
        keel::signal::SignalStatus::kOk) {
        return CommStatus::kInvalidArgument;
    }

    const f64 two_pi_fc = kTau * fc;
    const f64 inv_fs = 1.0 / fs;
    for (usize r = 0; r < y.size(); ++r) {
        const f64 ang = two_pi_fc * (as<f64>(r) * inv_fs) + init_phase;
        const f64 ca = keel::cos(ang);
        const f64 sa = keel::sin(ang);
        const f64 re = analytic[r].real();
        const f64 im = analytic[r].imag();
        out[r] = keel::atan2(im * ca - re * sa, re * ca + im * sa) / phase_dev;
    }
    return CommStatus::kOk;
}

CommStatus am_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 init_phase, f64 carrier_amp,
                  MutArrayRef<f64> out) noexcept {
    if (!(fs > 0.0) || !(fc > 0.0) || fs < 2.0 * fc || out.size() < x.size()) {
        return CommStatus::kInvalidArgument;
    }
    const f64 two_pi_fc = kTau * fc;
    const f64 inv_fs = 1.0 / fs;
    for (usize r = 0; r < x.size(); ++r) {
        const f64 t = as<f64>(r) * inv_fs;
        out[r] = (x[r] + carrier_amp) * keel::cos(two_pi_fc * t + init_phase);
    }
    return CommStatus::kOk;
}

CommStatus fm_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 freq_dev, f64 init_phase,
                  MutArrayRef<f64> out) noexcept {
    if (!(fs > 0.0) || fc < 0.0 || fs < 2.0 * fc || !(freq_dev > 0.0) || out.size() < x.size()) {
        return CommStatus::kInvalidArgument;
    }
    const f64 two_pi_fc = kTau * fc;
    const f64 two_pi_fd = kTau * freq_dev;
    const f64 inv_fs = 1.0 / fs;
    f64 acc = 0.0;
    for (usize r = 0; r < x.size(); ++r) {
        acc += x[r] * inv_fs;
        const f64 t = as<f64>(r) * inv_fs;
        out[r] = keel::cos(two_pi_fc * t + two_pi_fd * acc + init_phase);
    }
    return CommStatus::kOk;
}

CommStatus fm_demod(ArrayRef<f64> y, f64 fc, f64 fs, f64 freq_dev, f64 init_phase,
                    MutArrayRef<f64> out, Allocator* alloc) {
    if (!(fs > 0.0) || !(fc > 0.0) || fs < 2.0 * fc || !(freq_dev > 0.0) ||
        out.size() < y.size()) {
        return CommStatus::kInvalidArgument;
    }
    if (y.empty()) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> analytic(y.size(), c64(0.0, 0.0), scratch.allocator());
    if (keel::signal::hilbert(y, analytic.as_mut_ref(), scratch.allocator()) !=
        keel::signal::SignalStatus::kOk) {
        return CommStatus::kInvalidArgument;
    }

    const f64 two_pi_fc = kTau * fc;
    const f64 inv_fs = 1.0 / fs;
    const f64 scale = fs / (kTau * freq_dev);
    Array<f64> ph(y.size(), 0.0, scratch.allocator());
    for (usize r = 0; r < y.size(); ++r) {
        const f64 ang = two_pi_fc * (as<f64>(r) * inv_fs) + init_phase;
        const f64 ca = keel::cos(ang);
        const f64 sa = keel::sin(ang);
        const f64 re = analytic[r].real();
        const f64 im = analytic[r].imag();
        ph[r] = keel::atan2(im * ca - re * sa, re * ca + im * sa);
    }
    out[0] = 0.0;
    for (usize r = 1; r < y.size(); ++r) {
        f64 dlt = ph[r] - ph[r - 1u];
        dlt -= kTau * keel::round(dlt / kTau);
        out[r] = dlt * scale;
    }
    return CommStatus::kOk;
}

CommStatus ssb_mod(ArrayRef<f64> x, f64 fc, f64 fs, f64 init_phase, bool upper,
                   MutArrayRef<f64> out, Allocator* alloc) {
    if (!(fs > 0.0) || !(fc > 0.0) || fs <= 2.0 * fc || out.size() < x.size()) {
        return CommStatus::kInvalidArgument;
    }
    if (x.empty()) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> analytic(x.size(), c64(0.0, 0.0), scratch.allocator());
    if (keel::signal::hilbert(x, analytic.as_mut_ref(), scratch.allocator()) !=
        keel::signal::SignalStatus::kOk) {
        return CommStatus::kInvalidArgument;
    }

    const f64 two_pi_fc = kTau * fc;
    const f64 inv_fs = 1.0 / fs;
    const f64 sign = upper ? -1.0 : 1.0;
    for (usize r = 0; r < x.size(); ++r) {
        const f64 ang = two_pi_fc * (as<f64>(r) * inv_fs) + init_phase;
        out[r] = x[r] * keel::cos(ang) + sign * analytic[r].imag() * keel::sin(ang);
    }
    return CommStatus::kOk;
}

CommStatus msk_mod_diff(ArrayRef<u8> bits, i32 nsamp, f64 init_phase, MutArrayRef<c64> out,
                        Allocator* alloc) {
    if (nsamp <= 0) return CommStatus::kInvalidArgument;
    const usize ns = as<usize>(nsamp);
    const usize nout = bits.size() * ns;
    if (out.size() < nout) return CommStatus::kInvalidArgument;
    if (nout == 0) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> x_cum(bits.size() + 1u, 0.0, scratch.allocator());
    for (usize i = 0; i < bits.size(); ++i) {
        if (bits[i] != 0u && bits[i] != 1u) return CommStatus::kInvalidArgument;
        x_cum[i + 1u] = x_cum[i] + (2.0 * as<f64>(bits[i]) - 1.0);
    }

    const f64 inv_ns = 1.0 / as<f64>(nsamp);
    for (usize j = 0; j < nout; ++j) {
        const f64 t_fine = as<f64>(j) * inv_ns;
        const usize base = as<usize>(keel::floor(t_fine));
        const f64 frac = t_fine - as<f64>(base);
        const f64 interp = x_cum[base] + frac * (x_cum[base + 1u] - x_cum[base]);
        const f64 phase = 0.5 * kPi * interp + init_phase;
        out[j] = c64(keel::cos(phase), keel::sin(phase));
    }
    return CommStatus::kOk;
}

CommStatus msk_demod_diff(ArrayRef<c64> samples, i32 nsamp, f64 init_phase,
                          MutArrayRef<u8> out_bits, f64* phase_out) noexcept {
    if (nsamp <= 0 || samples.empty() || (samples.size() % as<usize>(nsamp)) != 0u) {
        return CommStatus::kInvalidArgument;
    }
    const usize ns = as<usize>(nsamp);
    const usize nsym = samples.size() / ns;
    if (out_bits.size() < nsym) return CommStatus::kInvalidArgument;

    i64 cum_sum = 0;
    for (usize k = 0; k < nsym; ++k) {
        const usize start = k * ns;
        f64 acc = 0.0;
        for (usize n = start + 1u; n < start + ns; ++n) {
            const c64 p = samples[n] * keel::conj(samples[n - 1u]);
            acc += keel::atan2(p.imag(), p.real());
        }
        const u8 bit = (acc > 0.0) ? 1u : 0u;
        out_bits[k] = bit;
        cum_sum += (bit != 0u) ? 1 : -1;
    }

    if (phase_out) {
        f64 ph = keel::fmod(init_phase + 0.5 * kPi * as<f64>(cum_sum), kTau);
        if (ph < 0.0) ph += kTau;
        *phase_out = ph;
    }
    return CommStatus::kOk;
}

} // namespace keel::comm
