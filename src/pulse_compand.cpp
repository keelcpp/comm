#include "keel/comm.h"

#include "keel/math.h"

namespace keel::comm {

namespace {

[[nodiscard]] f64 sinc_pi(f64 x) noexcept {
    if (keel::abs(x) < 1e-12) return 1.0;
    const f64 a = kPi * x;
    return keel::sin(a) / a;
}

} // namespace

CommStatus rcos_design(f64 beta, usize span, usize sps, RcosShape shape,
                       MutArrayRef<f64> out) noexcept {
    if (!(beta >= 0.0 && beta <= 1.0) || span == 0 || sps == 0) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = span * sps + 1;
    if (out.size() < n) {
        return CommStatus::kInvalidArgument;
    }

    const f64 dt = 1.0 / as<f64>(sps);
    const f64 half = 0.5 * as<f64>(n - 1);
    const bool rrc = (shape == RcosShape::kSqrt);

    f64 energy = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 t = (as<f64>(i) - half) * dt;
        f64 v = 0.0;
        if (!rrc) {
            const f64 term = 2.0 * beta * t;
            const f64 denom = 1.0 - term * term;
            if (keel::abs(denom) < 1e-12) {
                v = (kPi * 0.25) * sinc_pi(1.0 / (2.0 * beta));
            } else {
                v = sinc_pi(t) * keel::cos(kPi * beta * t) / denom;
            }
        } else {
            if (keel::abs(t) < 1e-12) {
                v = 1.0 + beta * (4.0 / kPi - 1.0);
            } else if (beta > 0.0 && keel::abs(keel::abs(t) - 1.0 / (4.0 * beta)) < 1e-12) {
                const f64 a = beta / kSqrt2;
                const f64 s = keel::sin(kPi / (4.0 * beta));
                const f64 c = keel::cos(kPi / (4.0 * beta));
                v = a * ((1.0 + 2.0 / kPi) * s + (1.0 - 2.0 / kPi) * c);
            } else {
                const f64 num = keel::cos((1.0 + beta) * kPi * t) +
                                keel::sin((1.0 - beta) * kPi * t) / (4.0 * beta * t);
                const f64 term = 4.0 * beta * t;
                const f64 den = 1.0 - term * term;
                v = (4.0 * beta / kPi) * num / den;
            }
        }
        out[i] = v;
        energy += v * v;
    }

    if (energy > 0.0) {
        const f64 inv = 1.0 / keel::sqrt(energy);
        for (usize i = 0; i < n; ++i) {
            out[i] *= inv;
        }
    }
    return CommStatus::kOk;
}

CommStatus gauss_design(f64 bt, usize span, usize sps, MutArrayRef<f64> out) noexcept {
    if (!(bt > 0.0) || !keel::is_finite(bt) || span == 0 || sps == 0) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = span * sps + 1;
    if (out.size() < n) {
        return CommStatus::kInvalidArgument;
    }

    const f64 centre = 0.5 * as<f64>(1 + n);
    const f64 alpha = keel::sqrt(kLn2 * 0.5) / bt;
    const f64 pre = keel::sqrt(kPi) / alpha;
    const f64 sps_f = as<f64>(sps);

    f64 sum = 0.0;
    for (usize k = 1; k <= n; ++k) {
        const f64 t = (as<f64>(k) - centre) / sps_f;
        const f64 arg = kPi * t / alpha;
        const f64 v = pre * keel::exp(-(arg * arg));
        out[k - 1] = v;
        sum += v;
    }
    if (sum > 0.0) {
        const f64 inv = 1.0 / sum;
        for (usize i = 0; i < n; ++i) {
            out[i] *= inv;
        }
    }
    return CommStatus::kOk;
}

CommStatus compand(ArrayRef<f64> in, f64 param, f64 peak_v, CompandMode mode,
                   MutArrayRef<f64> out) noexcept {
    if (!(param > 0.0) || !(peak_v > 0.0) || out.size() < in.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = in.size();
    switch (mode) {
        case CompandMode::kMuCompress: {
            const f64 log_mu = keel::log(1.0 + param);
            for (usize i = 0; i < n; ++i) {
                const f64 xi = in[i];
                const f64 mag = peak_v * keel::log(1.0 + param * keel::abs(xi) / peak_v) / log_mu;
                out[i] = (xi >= 0.0) ? mag : -mag;
            }
            break;
        }
        case CompandMode::kMuExpand: {
            const f64 log_mu = keel::log(1.0 + param);
            for (usize i = 0; i < n; ++i) {
                const f64 yi = in[i];
                const f64 mag = (peak_v / param) * (keel::exp(keel::abs(yi) / peak_v * log_mu) - 1.0);
                out[i] = (yi >= 0.0) ? mag : -mag;
            }
            break;
        }
        case CompandMode::kACompress: {
            const f64 one_log_a = 1.0 + keel::log(param);
            const f64 thr = 1.0 / param;
            for (usize i = 0; i < n; ++i) {
                const f64 xi = in[i];
                const f64 absx = keel::abs(xi);
                const f64 ratio = absx / peak_v;
                const f64 mag = (ratio <= thr)
                                    ? ((param * absx) / one_log_a)
                                    : (peak_v * (1.0 + keel::log(param * ratio)) / one_log_a);
                out[i] = (xi >= 0.0) ? mag : -mag;
            }
            break;
        }
        case CompandMode::kAExpand: {
            const f64 one_log_a = 1.0 + keel::log(param);
            const f64 thr = 1.0 / one_log_a;
            for (usize i = 0; i < n; ++i) {
                const f64 yi = in[i];
                const f64 absy = keel::abs(yi);
                const f64 ratio = absy / peak_v;
                const f64 mag = (ratio <= thr)
                                    ? (absy * one_log_a / param)
                                    : (peak_v * keel::exp(one_log_a * ratio - 1.0) / param);
                out[i] = (yi >= 0.0) ? mag : -mag;
            }
            break;
        }
    }
    return CommStatus::kOk;
}

} // namespace keel::comm
