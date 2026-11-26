#include "keel/comm.h"

#include "keel/math.h"

namespace keel::comm {

namespace {

[[nodiscard]] i32 symbol_to_idx(i32 s, i32 order, SymbolOrder mapping) noexcept {
    if (mapping == SymbolOrder::kBinary) {
        return s;
    }
    return as<i32>(from_gray(as<u32>(s))) % order;
}

[[nodiscard]] i32 idx_to_symbol(i32 k, i32 order, SymbolOrder mapping) noexcept {
    if (mapping == SymbolOrder::kBinary) {
        return k;
    }
    return as<i32>(to_gray(as<u32>(k))) % order;
}

[[nodiscard]] f64 pam_amp(i32 s, i32 order, SymbolOrder mapping) noexcept {
    const i32 k = (mapping == SymbolOrder::kBinary)
                      ? s
                      : as<i32>(from_gray(as<u32>(s)));
    return 2.0 * as<f64>(k) - as<f64>(order - 1);
}

[[nodiscard]] i32 clamp_rounded_i32(f64 v, i32 hi) noexcept {
    if (keel::is_nan(v)) return 0;
    const f64 r = keel::round(v);
    if (r <= 0.0) return 0;
    const f64 hi_f = as<f64>(hi);
    if (r >= hi_f) return hi;
    return as<i32>(r);
}

[[nodiscard]] i32 wrap_angle_to_idx(f64 angle, f64 dphi, i32 order) noexcept {
    if (!keel::is_finite(angle)) return 0;
    const f64 w = keel::fmod(angle, kTau);
    i32 k = as<i32>(keel::round(w / dphi));
    k = ((k % order) + order) % order;
    return k;
}

[[nodiscard]] i32 pam_amp_to_symbol(f64 amp, i32 order, SymbolOrder mapping) noexcept {
    const i32 k = clamp_rounded_i32(0.5 * (amp + as<f64>(order - 1)), order - 1);
    if (mapping == SymbolOrder::kBinary) {
        return k;
    }
    return as<i32>(to_gray(as<u32>(k)));
}

struct QamGrid {
    i32 ki = 1;
    i32 kq = 1;
};

[[nodiscard]] QamGrid factor_qam_grid(i32 order) noexcept {
    i32 k = as<i32>(keel::round(keel::sqrt(as<f64>(order))));
    while (k > 1 && (order % k) != 0) {
        --k;
    }
    return QamGrid{order / k, k};
}

} // namespace

CommStatus psk_mod(ArrayRef<i32> symbols, i32 order, f64 init_phase, SymbolOrder mapping,
                   MutArrayRef<c64> out) noexcept {
    if (order < 2 || !keel::is_finite(init_phase) || out.size() < symbols.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = symbols.size();
    for (usize i = 0; i < n; ++i) {
        if (symbols[i] < 0 || symbols[i] >= order) {
            return CommStatus::kInvalidArgument;
        }
    }
    const f64 dphi = kTau / as<f64>(order);
    for (usize i = 0; i < n; ++i) {
        const i32 k = symbol_to_idx(symbols[i], order, mapping);
        const f64 theta = dphi * as<f64>(k) + init_phase;
        out[i] = c64(keel::cos(theta), keel::sin(theta));
    }
    return CommStatus::kOk;
}

CommStatus psk_demod(ArrayRef<c64> samples, i32 order, f64 init_phase, SymbolOrder mapping,
                     MutArrayRef<i32> out) noexcept {
    if (order < 2 || !keel::is_finite(init_phase) || out.size() < samples.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = samples.size();
    const f64 dphi = kTau / as<f64>(order);
    for (usize i = 0; i < n; ++i) {
        const f64 angle = keel::arg(samples[i]) - init_phase;
        const i32 k = wrap_angle_to_idx(angle, dphi, order);
        out[i] = idx_to_symbol(k, order, mapping);
    }
    return CommStatus::kOk;
}

CommStatus dpsk_mod(ArrayRef<i32> symbols, i32 order, f64 phase_rot, SymbolOrder mapping,
                    MutArrayRef<c64> out) noexcept {
    if (order < 2 || !keel::is_finite(phase_rot) || out.size() < symbols.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = symbols.size();
    for (usize i = 0; i < n; ++i) {
        if (symbols[i] < 0 || symbols[i] >= order) {
            return CommStatus::kInvalidArgument;
        }
    }
    const f64 dphi = kTau / as<f64>(order);
    f64 phase = 0.0;
    for (usize i = 0; i < n; ++i) {
        const i32 k = symbol_to_idx(symbols[i], order, mapping);
        phase = keel::fmod(phase + phase_rot + dphi * as<f64>(k), kTau);
        out[i] = c64(keel::cos(phase), keel::sin(phase));
    }
    return CommStatus::kOk;
}

CommStatus dpsk_demod(ArrayRef<c64> samples, i32 order, f64 phase_rot, SymbolOrder mapping,
                      MutArrayRef<i32> out) noexcept {
    if (order < 2 || !keel::is_finite(phase_rot) || out.size() < samples.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = samples.size();
    const f64 dphi = kTau / as<f64>(order);
    f64 prev_phase = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 cur = keel::arg(samples[i]);
        const i32 k = wrap_angle_to_idx(cur - prev_phase - phase_rot, dphi, order);
        out[i] = idx_to_symbol(k, order, mapping);
        prev_phase = cur;
    }
    return CommStatus::kOk;
}

CommStatus pam_mod(ArrayRef<i32> symbols, i32 order, f64 init_phase, SymbolOrder mapping,
                   MutArrayRef<c64> out) noexcept {
    if (order < 2 || !keel::is_finite(init_phase) || out.size() < symbols.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = symbols.size();
    for (usize i = 0; i < n; ++i) {
        if (symbols[i] < 0 || symbols[i] >= order) {
            return CommStatus::kInvalidArgument;
        }
    }
    const c64 phase =
        (init_phase == 0.0) ? c64(1.0, 0.0) : c64(keel::cos(init_phase), keel::sin(init_phase));
    for (usize i = 0; i < n; ++i) {
        const f64 amp = pam_amp(symbols[i], order, mapping);
        out[i] = c64(amp, 0.0) * phase;
    }
    return CommStatus::kOk;
}

CommStatus pam_demod(ArrayRef<c64> samples, i32 order, f64 init_phase, SymbolOrder mapping,
                     MutArrayRef<i32> out) noexcept {
    if (order < 2 || !keel::is_finite(init_phase) || out.size() < samples.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = samples.size();
    const c64 inv = keel::polar(1.0, -init_phase);
    for (usize i = 0; i < n; ++i) {
        const c64 c = samples[i] * inv;
        out[i] = pam_amp_to_symbol(c.real(), order, mapping);
    }
    return CommStatus::kOk;
}

CommStatus qam_mod(ArrayRef<i32> symbols, i32 order, SymbolOrder mapping, bool unit_power,
                   MutArrayRef<c64> out) noexcept {
    if (order < 4 || out.size() < symbols.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = symbols.size();
    for (usize i = 0; i < n; ++i) {
        if (symbols[i] < 0 || symbols[i] >= order) {
            return CommStatus::kInvalidArgument;
        }
    }
    const QamGrid g = factor_qam_grid(order);
    f64 scale = 1.0;
    if (unit_power) {
        const f64 ki_f = as<f64>(g.ki);
        const f64 kq_f = as<f64>(g.kq);
        const f64 pavg = ((ki_f * ki_f - 1.0) + (kq_f * kq_f - 1.0)) / 3.0;
        scale = (pavg > 0.0) ? (1.0 / keel::sqrt(pavg)) : 1.0;
    }
    for (usize i = 0; i < n; ++i) {
        const i32 s = symbols[i];
        const i32 col = s / g.kq;
        const i32 row = s % g.kq;
        const i32 gcol = (mapping == SymbolOrder::kBinary)
                             ? col
                             : as<i32>(from_gray(as<u32>(col)));
        const i32 grow = (mapping == SymbolOrder::kBinary)
                             ? row
                             : as<i32>(from_gray(as<u32>(row)));
        const f64 in_phase = (2.0 * as<f64>(gcol) - as<f64>(g.ki - 1)) * scale;
        const f64 quad = (as<f64>(g.kq - 1) - 2.0 * as<f64>(grow)) * scale;
        out[i] = c64(in_phase, quad);
    }
    return CommStatus::kOk;
}

CommStatus qam_demod(ArrayRef<c64> samples, i32 order, SymbolOrder mapping, bool unit_power,
                     MutArrayRef<i32> out) noexcept {
    if (order < 4 || out.size() < samples.size()) {
        return CommStatus::kInvalidArgument;
    }
    const QamGrid g = factor_qam_grid(order);
    f64 scale = 1.0;
    if (unit_power) {
        const f64 ki_f = as<f64>(g.ki);
        const f64 kq_f = as<f64>(g.kq);
        const f64 pavg = ((ki_f * ki_f - 1.0) + (kq_f * kq_f - 1.0)) / 3.0;
        scale = (pavg > 0.0) ? (1.0 / keel::sqrt(pavg)) : 1.0;
    }
    const usize n = samples.size();
    for (usize i = 0; i < n; ++i) {
        const f64 in_phase = samples[i].real() / scale;
        const f64 quad = samples[i].imag() / scale;
        const i32 gcol = clamp_rounded_i32(0.5 * (in_phase + as<f64>(g.ki - 1)), g.ki - 1);
        const i32 grow = clamp_rounded_i32(0.5 * (as<f64>(g.kq - 1) - quad), g.kq - 1);
        const i32 col = (mapping == SymbolOrder::kBinary)
                            ? gcol
                            : as<i32>(to_gray(as<u32>(gcol)));
        const i32 row = (mapping == SymbolOrder::kBinary)
                            ? grow
                            : as<i32>(to_gray(as<u32>(grow)));
        out[i] = col * g.kq + row;
    }
    return CommStatus::kOk;
}

f64 modnorm(ArrayRef<c64> ref_constellation, NormMode mode, f64 target_power) noexcept {
    if (!keel::is_finite(target_power) || target_power < 0.0) {
        return Limits<f64>::quiet_nan();
    }
    const usize n = ref_constellation.size();
    if (n == 0) return 1.0;

    f64 scale = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 ar = keel::abs(ref_constellation[i].real());
        const f64 ai = keel::abs(ref_constellation[i].imag());
        if (!keel::is_finite(ar) || !keel::is_finite(ai)) {
            return Limits<f64>::quiet_nan();
        }
        const f64 m = keel::max(ar, ai);
        if (m > scale) scale = m;
    }
    if (scale == 0.0) return 1.0;
    const f64 inv_scale = 1.0 / scale;

    if (mode == NormMode::kAveragePower) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) {
            const f64 re = ref_constellation[i].real() * inv_scale;
            const f64 im = ref_constellation[i].imag() * inv_scale;
            s += re * re + im * im;
        }
        s /= as<f64>(n);
        return (s > 0.0) ? (keel::sqrt(target_power / s) * inv_scale) : 1.0;
    }
    f64 pk = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 re = ref_constellation[i].real() * inv_scale;
        const f64 im = ref_constellation[i].imag() * inv_scale;
        const f64 p = re * re + im * im;
        if (p > pk) pk = p;
    }
    return (pk > 0.0) ? (keel::sqrt(target_power / pk) * inv_scale) : 1.0;
}

} // namespace keel::comm
