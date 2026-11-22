#include "keel/comm.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::comm {

namespace {

inline bool octal_to_decimal(u64 oct, u64& out) noexcept {
    u64 result = 0ULL;
    u64 mult = 1ULL;
    while (oct > 0ULL) {
        const u64 digit = oct % 10ULL;
        if (digit > 7ULL) return false;
        result += digit * mult;
        mult *= 8ULL;
        oct /= 10ULL;
    }
    out = result;
    return true;
}

inline u32 bit_parity(u64 x) noexcept {
    u32 p = 0u;
    while (x != 0ULL) {
        p ^= 1u;
        x &= (x - 1ULL);
    }
    return p;
}

inline u32 popcount_u32(u32 x) noexcept {
    u32 c = 0u;
    while (x != 0u) {
        ++c;
        x &= (x - 1u);
    }
    return c;
}

} // namespace

CommStatus poly2trellis(i32 constraint_len, ArrayRef<u64> octal_generators, Trellis& out,
                        Allocator* alloc) {
    if (constraint_len < 1 || constraint_len > 25 || octal_generators.empty() ||
        octal_generators.size() > 30) {
        return CommStatus::kInvalidArgument;
    }

    const usize n = octal_generators.size();
    ArenaTemp scratch = get_scratch(alloc);
    Array<u64> g(n, 0ULL, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        if (!octal_to_decimal(octal_generators[i], g[i])) {
            return CommStatus::kInvalidArgument;
        }
    }

    const usize num_states = usize{1} << as<usize>(constraint_len - 1);
    const usize num_input = 2u;
    const usize num_output = usize{1} << n;

    out.num_input_symbols = num_input;
    out.num_output_symbols = num_output;
    out.num_states = num_states;
    out.next_states = Array<u32>(num_states * num_input, 0u, alloc);
    out.outputs = Array<u32>(num_states * num_input, 0u, alloc);

    const u64 state_mask = as<u64>(num_states - 1u);
    for (usize s = 0; s < num_states; ++s) {
        for (usize u = 0; u < num_input; ++u) {
            const u64 reg = (as<u64>(u) << as<usize>(constraint_len - 1)) | as<u64>(s);
            const u32 next_state = as<u32>((reg >> 1) & state_mask);
            u32 word = 0u;
            for (usize i = 0; i < n; ++i) {
                word = (word << 1) | bit_parity(reg & g[i]);
            }
            out.next_states[u * num_states + s] = next_state;
            out.outputs[u * num_states + s] = word;
        }
    }
    return CommStatus::kOk;
}

CommStatus convenc(ArrayRef<u8> msg_bits, ArrayRef<u32> next_states_col_major,
                   ArrayRef<u32> outputs_col_major, usize num_states, i32 out_bits_per_sym,
                   MutArrayRef<u8> out_bits) noexcept {
    if (num_states == 0 || out_bits_per_sym <= 0 || out_bits_per_sym > 31 ||
        next_states_col_major.size() < num_states * 2u ||
        outputs_col_major.size() < num_states * 2u) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = as<usize>(out_bits_per_sym);
    const usize l = msg_bits.size();
    if (out_bits.size() < l * n) {
        return CommStatus::kInvalidArgument;
    }

    usize state = 0u;
    usize oi = 0u;
    for (usize i = 0; i < l; ++i) {
        const usize bit = (msg_bits[i] != 0u) ? 1u : 0u;
        const usize idx = bit * num_states + state;
        const u32 out_word = outputs_col_major[idx];
        for (i32 j = out_bits_per_sym - 1; j >= 0; --j) {
            out_bits[oi++] = as<u8>((out_word >> as<u32>(j)) & 1u);
        }
        state = as<usize>(next_states_col_major[idx]);
        if (state >= num_states) return CommStatus::kInvalidArgument;
    }
    return CommStatus::kOk;
}

CommStatus vitdec_hard(ArrayRef<u8> code_bits, ArrayRef<u32> next_states_col_major,
                       ArrayRef<u32> outputs_col_major, usize num_states, i32 out_bits_per_sym,
                       bool terminated, MutArrayRef<u8> out_bits, Allocator* alloc) {
    if (num_states == 0 || out_bits_per_sym <= 0 || out_bits_per_sym > 31 ||
        next_states_col_major.size() < num_states * 2u ||
        outputs_col_major.size() < num_states * 2u) {
        return CommStatus::kInvalidArgument;
    }
    const usize n = as<usize>(out_bits_per_sym);
    if ((code_bits.size() % n) != 0u) {
        return CommStatus::kInvalidArgument;
    }
    const usize l = code_bits.size() / n;
    if (out_bits.size() < l) {
        return CommStatus::kInvalidArgument;
    }
    if (l == 0) return CommStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u32> rsym(l, 0u, scratch.allocator());
    for (usize t = 0; t < l; ++t) {
        u32 w = 0u;
        for (usize j = 0; j < n; ++j) {
            w = (w << 1) | ((code_bits[t * n + j] != 0u) ? 1u : 0u);
        }
        rsym[t] = w;
    }

    const u32 k_inf = Limits<u32>::kMax;
    Array<u32> metric(num_states, k_inf, scratch.allocator());
    Array<u32> new_metric(num_states, k_inf, scratch.allocator());
    metric[0] = 0u;
    Array<i32> prev_state(l * num_states, -1, scratch.allocator());
    Array<u8> dec_bit(l * num_states, 0u, scratch.allocator());

    for (usize t = 0; t < l; ++t) {
        for (usize s = 0; s < num_states; ++s) new_metric[s] = k_inf;
        for (usize s = 0; s < num_states; ++s) {
            if (metric[s] == k_inf) continue;
            for (usize u = 0; u < 2u; ++u) {
                const usize idx = u * num_states + s;
                const usize ns = as<usize>(next_states_col_major[idx]);
                if (ns >= num_states) return CommStatus::kInvalidArgument;
                const u32 ow = outputs_col_major[idx];
                const u32 cand = metric[s] + popcount_u32(rsym[t] ^ ow);
                if (cand < new_metric[ns]) {
                    new_metric[ns] = cand;
                    prev_state[t * num_states + ns] = as<i32>(s);
                    dec_bit[t * num_states + ns] = as<u8>(u);
                }
            }
        }
        keel::swap(metric, new_metric);
    }

    usize end_state = 0u;
    if (!terminated) {
        u32 best = metric[0];
        for (usize s = 1; s < num_states; ++s) {
            if (metric[s] < best) {
                best = metric[s];
                end_state = s;
            }
        }
    }

    usize st = end_state;
    for (usize ti = l; ti-- > 0;) {
        out_bits[ti] = dec_bit[ti * num_states + st];
        const i32 p = prev_state[ti * num_states + st];
        st = (p >= 0) ? as<usize>(p) : 0u;
    }
    return CommStatus::kOk;
}

} // namespace keel::comm
