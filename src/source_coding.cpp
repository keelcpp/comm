#include "keel/comm.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::comm {

namespace {

// ============================================================================
// Huffman Helpers (binary min-heap without STL)
// ============================================================================

struct HuffHeapNode {
    f64 prob;
    i32 id;
    i32 order;
};

inline bool huff_less(const HuffHeapNode& a, const HuffHeapNode& b) noexcept {
    if (a.prob != b.prob) return a.prob < b.prob;
    return a.order < b.order;
}

void heap_push(Array<HuffHeapNode>& heap, HuffHeapNode node) {
    heap.push_back(node);
    usize idx = heap.size() - 1u;
    while (idx > 0u) {
        const usize parent = (idx - 1u) / 2u;
        if (!huff_less(heap[idx], heap[parent])) break;
        keel::swap(heap[idx], heap[parent]);
        idx = parent;
    }
}

HuffHeapNode heap_pop(Array<HuffHeapNode>& heap) noexcept {
    const HuffHeapNode top = heap[0];
    heap[0] = heap[heap.size() - 1u];
    heap.pop_back();
    const usize n = heap.size();
    usize idx = 0u;
    while (true) {
        const usize left = 2u * idx + 1u;
        const usize right = left + 1u;
        usize smallest = idx;
        if (left < n && huff_less(heap[left], heap[smallest])) smallest = left;
        if (right < n && huff_less(heap[right], heap[smallest])) smallest = right;
        if (smallest == idx) break;
        keel::swap(heap[idx], heap[smallest]);
        idx = smallest;
    }
    return top;
}

struct HuffTreeNode {
    i32 parent = -1;
    u8 bit = 0u;
};

struct DecodeTrieNode {
    i32 left = -1;
    i32 right = -1;
    i32 sym = -1;
};

// ============================================================================
// Arithmetic Coding Bit Helpers
// ============================================================================

inline u64 bitget(u64 x, i32 p) noexcept {
    return (x >> as<u32>(p - 1)) & 1ULL;
}

inline u64 mask_n(u64 x, i32 n) noexcept {
    return x & ((n == 64) ? ~0ULL : ((1ULL << as<u32>(n)) - 1ULL));
}

// ============================================================================
// DPCM-opt Autocorrelation & Levinson-Durbin
// ============================================================================

void dpcm_autocorr(ArrayRef<f64> x, i32 ord, MutArrayRef<f64> r) noexcept {
    const usize n = x.size();
    for (i32 k = 0; k <= ord; ++k) {
        f64 acc = 0.0;
        const usize last = n - as<usize>(k);
        for (usize i = 0; i < last; ++i) {
            acc += x[i] * x[i + as<usize>(k)];
        }
        r[as<usize>(k)] = acc / as<f64>(n - 1u - as<usize>(k));
    }
}

void dpcm_levinson_durbin(ArrayRef<f64> r, i32 ord, MutArrayRef<f64> a, MutArrayRef<f64> prev) noexcept {
    for (i32 i = 0; i <= ord; ++i) a[as<usize>(i)] = 0.0;
    a[0] = 1.0;
    f64 d = r[0];
    for (i32 m = 0; m < ord; ++m) {
        f64 beta = 0.0;
        for (i32 j = 0; j <= m; ++j) {
            beta += a[as<usize>(j)] * r[as<usize>(m + 1 - j)];
        }
        const f64 k_ref = (d != 0.0) ? (-beta / d) : 0.0;
        for (i32 i = 0; i <= m; ++i) prev[as<usize>(i)] = a[as<usize>(i)];
        for (i32 i = 1; i <= m + 1; ++i) {
            a[as<usize>(i)] += k_ref * prev[as<usize>(m + 1 - i)];
        }
        d = (1.0 - k_ref * k_ref) * d;
    }
}

} // namespace

// ============================================================================
// Huffman Coding
// ============================================================================

CommStatus huffman_dict(ArrayRef<f64> probs, HuffmanDict& out, Allocator* alloc) {
    const usize k = probs.size();
    if (k == 0) return CommStatus::kInvalidArgument;

    f64 sum = 0.0;
    for (usize i = 0; i < k; ++i) {
        if (!(probs[i] >= 0.0 && probs[i] <= 1.0)) return CommStatus::kInvalidArgument;
        sum += probs[i];
    }
    if (keel::abs(sum - 1.0) > keel::sqrt(Limits<f64>::kEpsilon)) {
        return CommStatus::kInvalidArgument;
    }

    out.offsets = Array<u32>(k + 1u, 0u, alloc);
    out.bits = Array<u8>(alloc);
    out.avg_len = 0.0;

    if (k == 1u) {
        out.offsets[0] = 0u;
        out.offsets[1] = 1u;
        out.bits.push_back(0u);
        out.avg_len = 1.0;
        return CommStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<HuffTreeNode> tree(2u * k - 1u, HuffTreeNode{}, scratch.allocator());
    Array<HuffHeapNode> heap(scratch.allocator());
    heap.reserve(k);

    i32 order_seed = 0;
    for (usize i = 0; i < k; ++i) {
        heap_push(heap, {probs[i], as<i32>(i), order_seed++});
    }

    i32 next_id = as<i32>(k);
    while (heap.size() > 1u) {
        const HuffHeapNode a = heap_pop(heap);
        const HuffHeapNode b = heap_pop(heap);
        tree[as<usize>(a.id)].parent = next_id;
        tree[as<usize>(a.id)].bit = 0u;
        tree[as<usize>(b.id)].parent = next_id;
        tree[as<usize>(b.id)].bit = 1u;
        heap_push(heap, {a.prob + b.prob, next_id, order_seed++});
        ++next_id;
    }

    Array<u8> rev_bits(scratch.allocator());
    f64 avglen = 0.0;
    for (usize i = 0; i < k; ++i) {
        rev_bits.clear();
        i32 cur = as<i32>(i);
        while (tree[as<usize>(cur)].parent != -1) {
            rev_bits.push_back(tree[as<usize>(cur)].bit);
            cur = tree[as<usize>(cur)].parent;
        }
        out.offsets[i] = as<u32>(out.bits.size());
        for (usize j = rev_bits.size(); j-- > 0;) {
            out.bits.push_back(rev_bits[j]);
        }
        avglen += probs[i] * as<f64>(rev_bits.size());
    }
    out.offsets[k] = as<u32>(out.bits.size());
    out.avg_len = avglen;
    return CommStatus::kOk;
}

CommStatus huffman_encode(ArrayRef<usize> sym_indices, ArrayRef<u32> offsets,
                          ArrayRef<u8> dict_bits, Array<u8>& out_bits, Allocator* alloc) {
    if (offsets.size() < 2u) return CommStatus::kInvalidArgument;
    const usize k = offsets.size() - 1u;
    if (offsets[k] > dict_bits.size()) return CommStatus::kInvalidArgument;

    usize total_bits = 0u;
    for (usize i = 0; i < sym_indices.size(); ++i) {
        const usize s = sym_indices[i];
        if (s >= k || offsets[s + 1u] < offsets[s]) return CommStatus::kInvalidArgument;
        total_bits += as<usize>(offsets[s + 1u] - offsets[s]);
    }

    out_bits = Array<u8>(alloc);
    out_bits.reserve(total_bits);
    for (usize i = 0; i < sym_indices.size(); ++i) {
        const usize s = sym_indices[i];
        for (u32 pos = offsets[s]; pos < offsets[s + 1u]; ++pos) {
            out_bits.push_back(dict_bits[as<usize>(pos)] & 1u);
        }
    }
    return CommStatus::kOk;
}

CommStatus huffman_decode(ArrayRef<u8> bits, ArrayRef<u32> offsets, ArrayRef<u8> dict_bits,
                          Array<usize>& out_sym_indices, Allocator* alloc) {
    if (offsets.size() < 2u) return CommStatus::kInvalidArgument;
    const usize k = offsets.size() - 1u;
    if (offsets[k] > dict_bits.size()) return CommStatus::kInvalidArgument;

    ArenaTemp scratch = get_scratch(alloc);
    Array<DecodeTrieNode> nodes(scratch.allocator());
    nodes.push_back(DecodeTrieNode{});

    for (usize s = 0; s < k; ++s) {
        if (offsets[s + 1u] < offsets[s]) return CommStatus::kInvalidArgument;
        i32 cur = 0;
        for (u32 pos = offsets[s]; pos < offsets[s + 1u]; ++pos) {
            const u8 bit = dict_bits[as<usize>(pos)];
            if (bit != 0u && bit != 1u) return CommStatus::kInvalidArgument;
            i32 child = (bit == 0u) ? nodes[as<usize>(cur)].left : nodes[as<usize>(cur)].right;
            if (child < 0) {
                nodes.push_back(DecodeTrieNode{});
                child = as<i32>(nodes.size() - 1u);
                if (bit == 0u) {
                    nodes[as<usize>(cur)].left = child;
                } else {
                    nodes[as<usize>(cur)].right = child;
                }
            }
            cur = child;
        }
        nodes[as<usize>(cur)].sym = as<i32>(s);
    }

    out_sym_indices = Array<usize>(alloc);
    i32 cur = 0;
    for (usize i = 0; i < bits.size(); ++i) {
        const u8 b = bits[i];
        if (b != 0u && b != 1u) return CommStatus::kInvalidArgument;
        const i32 next = (b == 0u) ? nodes[as<usize>(cur)].left : nodes[as<usize>(cur)].right;
        if (next < 0) return CommStatus::kDecodeFailed;
        cur = next;
        if (nodes[as<usize>(cur)].sym >= 0) {
            out_sym_indices.push_back(as<usize>(nodes[as<usize>(cur)].sym));
            cur = 0;
        }
    }
    if (cur != 0) return CommStatus::kDecodeFailed;
    return CommStatus::kOk;
}

// ============================================================================
// Arithmetic Coding (Sayood adaptive-precision integer implementation)
// ============================================================================

CommStatus arith_encode(ArrayRef<usize> seq_1based, ArrayRef<u64> counts, Array<u8>& out_bits,
                        Allocator* alloc) {
    const usize k = counts.size();
    if (k == 0) return CommStatus::kInvalidArgument;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u64> cum(k + 1u, 0ULL, scratch.allocator());
    for (usize i = 0; i < k; ++i) {
        if (counts[i] == 0ULL) return CommStatus::kInvalidArgument;
        cum[i + 1u] = cum[i] + counts[i];
    }
    const u64 total = cum[k];
    const i32 n = as<i32>(keel::ceil(keel::log2(as<f64>(total)))) + 2;
    if (n <= 2 || n >= 63) return CommStatus::kInvalidArgument;

    const u64 top_mask = 1ULL << as<u32>(n - 1);
    u64 dec_low = 0ULL;
    u64 dec_up = (1ULL << as<u32>(n)) - 1ULL;
    u64 e3_count = 0ULL;

    out_bits = Array<u8>(alloc);
    out_bits.reserve(seq_1based.size() * (as<usize>(keel::ceil(keel::log2(as<f64>(k))) + 2.0)) +
                     as<usize>(n));

    for (usize idx = 0; idx < seq_1based.size(); ++idx) {
        const usize s = seq_1based[idx];
        if (s == 0u || s > k) return CommStatus::kInvalidArgument;

        const u64 span = dec_up - dec_low + 1ULL;
        const u64 low_new = dec_low + (span * cum[s - 1u]) / total;
        dec_up = dec_low + (span * cum[s]) / total - 1ULL;
        dec_low = low_new;

        for (;;) {
            if (bitget(dec_low, n) == bitget(dec_up, n)) {
                const u64 b = bitget(dec_low, n);
                out_bits.push_back(as<u8>(b));
                dec_low = (dec_low << 1);
                dec_up = (dec_up << 1) + 1ULL;
                if (e3_count > 0ULL) {
                    for (u64 i = 0; i < e3_count; ++i) {
                        out_bits.push_back(as<u8>(b ^ 1ULL));
                    }
                    e3_count = 0ULL;
                }
                dec_low = mask_n(dec_low, n);
                dec_up = mask_n(dec_up, n);
            } else if (bitget(dec_low, n - 1) == 1ULL && bitget(dec_up, n - 1) == 0ULL) {
                dec_low = mask_n(dec_low << 1, n) ^ top_mask;
                dec_up = mask_n((dec_up << 1) + 1ULL, n) ^ top_mask;
                ++e3_count;
            } else {
                break;
            }
        }
    }

    Array<u8> bin_low(as<usize>(n), 0u, scratch.allocator());
    for (i32 i = 0; i < n; ++i) {
        bin_low[as<usize>(i)] = as<u8>((dec_low >> as<u32>(n - 1 - i)) & 1ULL);
    }
    if (e3_count == 0ULL) {
        for (i32 i = 0; i < n; ++i) out_bits.push_back(bin_low[as<usize>(i)]);
    } else {
        const u8 b = bin_low[0];
        out_bits.push_back(b);
        for (u64 i = 0; i < e3_count; ++i) out_bits.push_back(as<u8>(b ^ 1u));
        for (i32 i = 1; i < n; ++i) out_bits.push_back(bin_low[as<usize>(i)]);
    }
    return CommStatus::kOk;
}

CommStatus arith_decode(ArrayRef<u8> code_bits, ArrayRef<u64> counts, usize len,
                        MutArrayRef<usize> out_seq_1based, Allocator* alloc) {
    const usize k = counts.size();
    if (k == 0 || out_seq_1based.size() < len) return CommStatus::kInvalidArgument;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u64> cum(k + 1u, 0ULL, scratch.allocator());
    for (usize i = 0; i < k; ++i) {
        if (counts[i] == 0ULL) return CommStatus::kInvalidArgument;
        cum[i + 1u] = cum[i] + counts[i];
    }
    const u64 total = cum[k];
    const i32 n = as<i32>(keel::ceil(keel::log2(as<f64>(total)))) + 2;
    if (n <= 2 || n >= 63 || code_bits.size() < as<usize>(n)) {
        return CommStatus::kInvalidArgument;
    }

    const u64 top_mask = 1ULL << as<u32>(n - 1);
    u64 dec_low = 0ULL;
    u64 dec_up = (1ULL << as<u32>(n)) - 1ULL;
    u64 dec_tag = 0ULL;
    for (i32 i = 0; i < n; ++i) {
        const u8 b = code_bits[as<usize>(i)];
        if (b != 0u && b != 1u) return CommStatus::kInvalidArgument;
        dec_tag = (dec_tag << 1) | as<u64>(b);
    }

    auto pick = [&](u64 value) -> usize {
        if (value >= cum[k]) return k;
        for (usize i = k; i >= 1u; --i) {
            if (cum[i - 1u] <= value && value < cum[i]) return i;
        }
        return 1u;
    };

    usize bit_pos = as<usize>(n);
    for (usize i = 0; i < len; ++i) {
        const u64 span = dec_up - dec_low + 1ULL;
        const u64 dec_tag_new = ((dec_tag - dec_low + 1ULL) * total - 1ULL) / span;
        const usize ptr = pick(dec_tag_new);
        out_seq_1based[i] = ptr;

        const u64 low_new = dec_low + (span * cum[ptr - 1u]) / total;
        dec_up = dec_low + (span * cum[ptr]) / total - 1ULL;
        dec_low = low_new;

        for (;;) {
            const bool e1e2 = (bitget(dec_low, n) == bitget(dec_up, n));
            const bool e3 = (bitget(dec_low, n - 1) == 1ULL && bitget(dec_up, n - 1) == 0ULL);
            if (!e1e2 && !e3) break;
            if (bit_pos >= code_bits.size()) break;
            const u8 cb = code_bits[bit_pos++];
            if (cb != 0u && cb != 1u) return CommStatus::kInvalidArgument;
            dec_low = mask_n(dec_low << 1, n);
            dec_up = mask_n((dec_up << 1) + 1ULL, n);
            dec_tag = mask_n((dec_tag << 1) + as<u64>(cb), n);
            if (!e1e2) {
                dec_low ^= top_mask;
                dec_up ^= top_mask;
                dec_tag ^= top_mask;
            }
        }
    }
    return CommStatus::kOk;
}

// ============================================================================
// Scalar Quantization, Lloyd-Max & DPCM
// ============================================================================

CommStatus quantiz_index(ArrayRef<f64> sig, ArrayRef<f64> partition,
                         MutArrayRef<usize> out_idx) noexcept {
    if (out_idx.size() < sig.size()) return CommStatus::kInvalidArgument;
    const usize m = partition.size();
    for (usize i = 0; i < sig.size(); ++i) {
        const f64 xi = sig[i];
        usize cnt = 0u;
        while (cnt < m && partition[cnt] < xi) ++cnt;
        out_idx[i] = cnt;
    }
    return CommStatus::kOk;
}

CommStatus quantiz(ArrayRef<f64> sig, ArrayRef<f64> partition, ArrayRef<f64> codebook,
                   MutArrayRef<usize> out_idx, MutArrayRef<f64> out_quant,
                   f64& out_distor) noexcept {
    if (codebook.size() != partition.size() + 1u || out_idx.size() < sig.size() ||
        out_quant.size() < sig.size()) {
        return CommStatus::kInvalidArgument;
    }
    const CommStatus st = quantiz_index(sig, partition, out_idx);
    if (st != CommStatus::kOk) return st;
    f64 sse = 0.0;
    for (usize i = 0; i < sig.size(); ++i) {
        const f64 cv = codebook[out_idx[i]];
        out_quant[i] = cv;
        const f64 d = sig[i] - cv;
        sse += d * d;
    }
    out_distor = sig.empty() ? 0.0 : (sse / as<f64>(sig.size()));
    return CommStatus::kOk;
}

CommStatus lloyds(ArrayRef<f64> training, ArrayRef<f64> init_codebook, usize init_k, f64 tol,
                  LloydsResult& out, Allocator* alloc) {
    if (training.empty()) return CommStatus::kInvalidArgument;

    f64 min_train = training[0];
    f64 max_train = training[0];
    for (usize i = 1; i < training.size(); ++i) {
        if (training[i] < min_train) min_train = training[i];
        if (training[i] > max_train) max_train = training[i];
    }
    if (max_train <= min_train) return CommStatus::kInvalidArgument;

    Array<f64> codebook(alloc);
    if (!init_codebook.empty()) {
        codebook = Array<f64>::from(init_codebook, alloc);
        keel::sort(codebook.as_mut_ref());
    } else if (init_k > 0u) {
        codebook.resize(init_k);
        const f64 step = (max_train - min_train) / as<f64>(init_k);
        for (usize i = 0; i < init_k; ++i) {
            codebook[i] = min_train + step * 0.5 + step * as<f64>(i);
        }
    } else {
        return CommStatus::kInvalidArgument;
    }

    const usize k = codebook.size();
    if (tol <= 0.0) tol = 1e-7;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> partition(k > 0u ? k - 1u : 0u, 0.0, alloc);
    Array<usize> bins(training.size(), 0u, scratch.allocator());

    auto update_partition = [&]() {
        for (usize i = 0; i + 1u < k; ++i) {
            partition[i] = 0.5 * (codebook[i] + codebook[i + 1u]);
        }
        keel::sort(partition.as_mut_ref());
    };
    auto classify = [&]() {
        (void)quantiz_index(training, partition.as_ref(), bins.as_mut_ref());
    };
    auto calc_distor = [&]() -> f64 {
        f64 sse = 0.0;
        for (usize i = 0; i < training.size(); ++i) {
            const f64 d = training[i] - codebook[bins[i]];
            sse += d * d;
        }
        return sse / as<f64>(training.size());
    };

    update_partition();
    classify();
    f64 distor = calc_distor();
    f64 last_distor = 0.0;
    const f64 ter_cond2 = Limits<f64>::kEpsilon * keel::abs(max_train);
    f64 rel_distor = (distor > ter_cond2) ? (keel::abs(distor - last_distor) / distor) : distor;

    while (rel_distor > tol && rel_distor > ter_cond2) {
        for (usize i = 0; i < k; ++i) {
            f64 sum = 0.0;
            usize cnt = 0u;
            for (usize j = 0; j < training.size(); ++j) {
                if (bins[j] == i) {
                    sum += training[j];
                    ++cnt;
                }
            }
            if (cnt > 0u) {
                codebook[i] = sum / as<f64>(cnt);
            } else if (i == 0u) {
                if (partition.empty()) {
                    codebook[i] = 0.5 * (max_train + min_train);
                } else {
                    f64 s = 0.0;
                    usize c = 0u;
                    for (usize j = 0; j < training.size(); ++j) {
                        if (training[j] <= partition[0]) {
                            s += training[j];
                            ++c;
                        }
                    }
                    codebook[i] = (c > 0u) ? (s / as<f64>(c)) : (0.5 * (partition[0] + min_train));
                }
            } else if (i == k - 1u) {
                f64 s = 0.0;
                usize c = 0u;
                for (usize j = 0; j < training.size(); ++j) {
                    if (training[j] >= partition[i - 1u]) {
                        s += training[j];
                        ++c;
                    }
                }
                codebook[i] = (c > 0u) ? (s / as<f64>(c)) : (0.5 * (max_train + partition[i - 1u]));
            } else {
                f64 s = 0.0;
                usize c = 0u;
                for (usize j = 0; j < training.size(); ++j) {
                    if (training[j] >= partition[i - 1u] && training[j] <= partition[i]) {
                        s += training[j];
                        ++c;
                    }
                }
                codebook[i] =
                    (c > 0u) ? (s / as<f64>(c)) : (0.5 * (partition[i] + partition[i - 1u]));
            }
        }
        update_partition();
        classify();
        last_distor = distor;
        distor = calc_distor();
        rel_distor = (distor > ter_cond2) ? (keel::abs(distor - last_distor) / distor) : distor;
    }

    out.partition = keel::move(partition);
    out.codebook = keel::move(codebook);
    out.distortion = distor;
    out.rel_distortion = rel_distor;
    return CommStatus::kOk;
}

CommStatus dpcm_encode(ArrayRef<f64> sig, ArrayRef<f64> codebook, ArrayRef<f64> partition,
                       ArrayRef<f64> predictor, MutArrayRef<usize> out_idx,
                       MutArrayRef<f64> out_qerr, Allocator* alloc) {
    if (codebook.size() != partition.size() + 1u || predictor.size() < 2u ||
        out_idx.size() < sig.size() || out_qerr.size() < sig.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize m = predictor.size() - 1u;
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> state(m, 0.0, scratch.allocator());

    for (usize i = 0; i < sig.size(); ++i) {
        f64 pred = 0.0;
        for (usize k = 0; k < m; ++k) pred += predictor[k + 1u] * state[k];
        const f64 err = sig[i] - pred;
        usize bin = 0u;
        while (bin < partition.size() && partition[bin] < err) ++bin;
        const f64 q = codebook[bin];
        out_idx[i] = bin;
        out_qerr[i] = q;
        const f64 recon = q + pred;
        for (usize k = m; k-- > 1u;) state[k] = state[k - 1u];
        if (m > 0u) state[0] = recon;
    }
    return CommStatus::kOk;
}

CommStatus dpcm_decode(ArrayRef<usize> idx, ArrayRef<f64> codebook, ArrayRef<f64> predictor,
                       MutArrayRef<f64> out_sig, MutArrayRef<f64> out_qerr, Allocator* alloc) {
    if (predictor.size() < 2u || out_sig.size() < idx.size() || out_qerr.size() < idx.size()) {
        return CommStatus::kInvalidArgument;
    }
    const usize k_len = codebook.size();
    for (usize i = 0; i < idx.size(); ++i) {
        if (idx[i] >= k_len) return CommStatus::kInvalidArgument;
        out_qerr[i] = codebook[idx[i]];
    }
    const usize m = predictor.size() - 1u;
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> state(m, 0.0, scratch.allocator());

    for (usize i = 0; i < idx.size(); ++i) {
        f64 pred = 0.0;
        for (usize k = 0; k < m; ++k) pred += predictor[k + 1u] * state[k];
        const f64 s = out_qerr[i] + pred;
        out_sig[i] = s;
        for (usize k = m; k-- > 1u;) state[k] = state[k - 1u];
        if (m > 0u) state[0] = s;
    }
    return CommStatus::kOk;
}

CommStatus dpcm_opt(ArrayRef<f64> training, i32 order, ArrayRef<f64> init_codebook, usize init_k,
                    DpcmOptResult& out, Allocator* alloc) {
    if (order < 1 || training.size() < as<usize>(order) + 3u) {
        return CommStatus::kInvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    const usize ord_u = as<usize>(order);
    Array<f64> r(ord_u + 1u, 0.0, scratch.allocator());
    Array<f64> a(ord_u + 1u, 0.0, scratch.allocator());
    Array<f64> prev(ord_u + 1u, 0.0, scratch.allocator());

    dpcm_autocorr(training, order, r.as_mut_ref());
    dpcm_levinson_durbin(r.as_ref(), order, a.as_mut_ref(), prev.as_mut_ref());

    out.predictor = Array<f64>(ord_u + 1u, 0.0, alloc);
    out.predictor[0] = 0.0;
    for (usize k = 1u; k <= ord_u; ++k) out.predictor[k] = -a[k];
    out.partition = Array<f64>(alloc);
    out.codebook = Array<f64>(alloc);

    if (init_codebook.empty() && init_k == 0u) {
        return CommStatus::kOk;
    }

    const usize res_len = training.size() - ord_u;
    Array<f64> residual(res_len, 0.0, scratch.allocator());
    for (usize i = ord_u; i < training.size(); ++i) {
        f64 pred = 0.0;
        for (usize k = 1u; k <= ord_u; ++k) {
            pred += out.predictor[k] * training[i - k];
        }
        residual[i - ord_u] = training[i] - pred;
    }

    LloydsResult ll;
    const CommStatus st = lloyds(residual.as_ref(), init_codebook, init_k, 1e-7, ll, alloc);
    if (st != CommStatus::kOk) return st;
    out.partition = keel::move(ll.partition);
    out.codebook = keel::move(ll.codebook);
    return CommStatus::kOk;
}

} // namespace keel::comm
