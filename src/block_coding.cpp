#include "keel/comm.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"

namespace keel::comm {

namespace {

inline u8 col_bit(ArrayRef<u8> mat, usize rows, usize r, usize c) noexcept {
    return (mat[c * rows + r] != 0u) ? 1u : 0u;
}

inline i32 poly_deg(ArrayRef<u8> a) noexcept {
    for (i32 i = as<i32>(a.size()) - 1; i >= 0; --i) {
        if ((a[as<usize>(i)] & 1u) != 0u) return i;
    }
    return -1;
}

Array<u8> poly_mod(ArrayRef<u8> a_in, ArrayRef<u8> b, Allocator* alloc) {
    Array<u8> a = Array<u8>::from(a_in, alloc);
    const i32 db = poly_deg(b);
    for (usize i = 0; i < a.size(); ++i) a[i] &= 1u;
    for (i32 da = poly_deg(a.as_ref()); da >= db && da >= 0; da = poly_deg(a.as_ref())) {
        for (i32 j = 0; j <= db; ++j) {
            a[as<usize>(j + da - db)] ^= (b[as<usize>(j)] & 1u);
        }
    }
    a.resize(db > 0 ? as<usize>(db) : 0u);
    return a;
}

struct PolyQr {
    Array<u8> q;
    Array<u8> r;
};

PolyQr poly_div_mod(ArrayRef<u8> a_in, ArrayRef<u8> b, Allocator* alloc) {
    Array<u8> a = Array<u8>::from(a_in, alloc);
    const i32 db = poly_deg(b);
    for (usize i = 0; i < a.size(); ++i) a[i] &= 1u;
    const i32 da0 = poly_deg(a.as_ref());
    Array<u8> q(da0 >= db ? as<usize>(da0 - db + 1) : 1u, 0u, alloc);
    for (i32 da = da0; da >= db && da >= 0; da = poly_deg(a.as_ref())) {
        q[as<usize>(da - db)] ^= 1u;
        for (i32 j = 0; j <= db; ++j) {
            a[as<usize>(j + da - db)] ^= (b[as<usize>(j)] & 1u);
        }
    }
    a.resize(db > 0 ? as<usize>(db) : 1u);
    return {keel::move(q), keel::move(a)};
}

Array<u8> shift_matrix_col_major(ArrayRef<u8> row_in, usize rows, usize n, Allocator* alloc) {
    Array<u8> row = Array<u8>::from(row_in, alloc);
    Array<u8> out(rows * n, 0u, alloc);
    for (usize r = 0; r < rows; ++r) {
        for (usize c = 0; c < n; ++c) {
            out[c * rows + r] = row[c] & 1u;
        }
        const u8 last = row[n - 1];
        for (usize c = n - 1; c > 0; --c) row[c] = row[c - 1];
        row[0] = last;
    }
    return out;
}

bool next_combination(MutArrayRef<usize> c, usize n) noexcept {
    const usize w = c.size();
    if (w == 0) return false;
    usize i = w - 1;
    while (true) {
        if (c[i] < n - w + i) {
            ++c[i];
            for (usize j = i + 1; j < w; ++j) c[j] = c[j - 1] + 1;
            return true;
        }
        if (i == 0) return false;
        --i;
    }
}

} // namespace

u64 default_prim_poly(i32 m) noexcept {
    switch (m) {
        case 2: return 0x7ULL;
        case 3: return 0xBULL;
        case 4: return 0x13ULL;
        case 5: return 0x25ULL;
        case 6: return 0x43ULL;
        case 7: return 0x89ULL;
        case 8: return 0x11DULL;
        case 9: return 0x211ULL;
        case 10: return 0x409ULL;
        case 11: return 0x805ULL;
        case 12: return 0x1053ULL;
        case 13: return 0x201BULL;
        case 14: return 0x4443ULL;
        case 15: return 0x8003ULL;
        case 16: return 0x1100BULL;
        default: return 0ULL;
    }
}

CommStatus gen2par(ArrayRef<u8> mat_col_major, usize rows, usize cols, Array<u8>& out_col_major,
                   Allocator* alloc) {
    if (rows == 0 || cols == 0 || rows >= cols || mat_col_major.size() / cols < rows) {
        return CommStatus::kInvalidArgument;
    }
    const usize pcols = cols - rows;

    bool last_is_i = true;
    for (usize i = 0; i < rows && last_is_i; ++i) {
        for (usize t = 0; t < rows; ++t) {
            if (col_bit(mat_col_major, rows, i, pcols + t) != (i == t ? 1u : 0u)) {
                last_is_i = false;
                break;
            }
        }
    }

    bool first_is_i = true;
    for (usize i = 0; i < rows && first_is_i; ++i) {
        for (usize t = 0; t < rows; ++t) {
            if (col_bit(mat_col_major, rows, i, t) != (i == t ? 1u : 0u)) {
                first_is_i = false;
                break;
            }
        }
    }

    if (!last_is_i && !first_is_i) {
        return CommStatus::kNotSystematic;
    }

    out_col_major = Array<u8>(pcols * cols, 0u, alloc);
    if (last_is_i) {
        for (usize i = 0; i < pcols; ++i) {
            for (usize j = 0; j < pcols; ++j) {
                out_col_major[j * pcols + i] = (i == j) ? 1u : 0u;
            }
            for (usize t = 0; t < rows; ++t) {
                out_col_major[(pcols + t) * pcols + i] = col_bit(mat_col_major, rows, t, i);
            }
        }
    } else {
        for (usize i = 0; i < pcols; ++i) {
            for (usize t = 0; t < rows; ++t) {
                out_col_major[t * pcols + i] = col_bit(mat_col_major, rows, t, rows + i);
            }
            for (usize j = 0; j < pcols; ++j) {
                out_col_major[(rows + j) * pcols + i] = (i == j) ? 1u : 0u;
            }
        }
    }
    return CommStatus::kOk;
}

CommStatus hammgen(i32 m, u64 prim_mask, BlockCodeMatrices& out, Allocator* alloc) {
    if (m < 2 || m > 24) {
        return CommStatus::kInvalidArgument;
    }
    const u64 pmask = (prim_mask == 0ULL) ? default_prim_poly(m) : prim_mask;
    if (pmask == 0ULL || (pmask & 1ULL) == 0ULL || (pmask & (1ULL << as<u32>(m))) == 0ULL ||
        (pmask >> as<u32>(m + 1)) != 0ULL) {
        return CommStatus::kInvalidArgument;
    }

    const usize mu = as<usize>(m);
    const usize n = (usize{1} << mu) - 1u;
    const usize k = n - mu;

    out.h = Array<u8>(mu * n, 0u, alloc);
    const u64 top_bit = 1ULL << m;
    u64 r = 1ULL;
    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < mu; ++j) {
            out.h[i * mu + j] = as<u8>((r >> j) & 1ULL);
        }
        r <<= 1;
        if ((r & top_bit) != 0ULL) r ^= pmask;
    }

    const CommStatus st = gen2par(out.h.as_ref(), mu, n, out.g, alloc);
    if (st != CommStatus::kOk) return st;
    out.n = n;
    out.k = k;
    return CommStatus::kOk;
}

CommStatus cyclpoly(i32 n, i32 k, CyclPolyMode mode, CyclPolyResult& out, Allocator* alloc) {
    if (n < 1 || k < 1 || k > n) {
        return CommStatus::kInvalidArgument;
    }
    const i32 m = n - k;
    if (m > 25) {
        return CommStatus::kInvalidArgument;
    }
    out.rows_col_major = Array<u8>(alloc);
    out.num_polys = 0;
    out.poly_len = as<usize>(m) + 1u;

    if (m == 0) {
        out.rows_col_major.push_back(1u);
        out.num_polys = 1;
        return CommStatus::kOk;
    }
    if (m == 1) {
        out.rows_col_major.push_back(1u);
        out.rows_col_major.push_back(1u);
        out.num_polys = 1;
        return CommStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> pp(as<usize>(n) + 1u, 0u, scratch.allocator());
    pp[0] = 1u;
    pp[as<usize>(n)] = 1u;

    const u64 nn = (1ULL << (m - 1)) - 1ULL;
    const usize plen = as<usize>(m) + 1u;
    Array<u8> cand(plen, 0u, scratch.allocator());

    auto fill_cand = [&](u64 idx, bool msb_first) {
        cand[0] = 1u;
        cand[as<usize>(m)] = 1u;
        for (i32 p = 0; p < m - 1; ++p) {
            const i32 shift = msb_first ? (m - 2 - p) : p;
            cand[as<usize>(1 + p)] = as<u8>((idx >> shift) & 1ULL);
        }
    };
    auto divides = [&]() {
        const Array<u8> rem = poly_mod(pp.as_ref(), cand.as_ref(), scratch.allocator());
        return poly_deg(rem.as_ref()) < 0;
    };
    auto count_terms = [](ArrayRef<u8> t) -> i32 {
        i32 s = 0;
        for (usize i = 0; i < t.size(); ++i) s += as<i32>(t[i] & 1u);
        return s;
    };

    if (mode == CyclPolyMode::kFirst) {
        for (u64 i = 0; i <= nn; ++i) {
            fill_cand(i, true);
            if (divides()) {
                out.rows_col_major = Array<u8>::from(cand.as_ref(), alloc);
                out.num_polys = 1;
                return CommStatus::kOk;
            }
        }
        return CommStatus::kOk;
    }

    if (mode == CyclPolyMode::kMin) {
        for (i32 j = 2; j <= m + 1; ++j) {
            for (u64 i = 0; i <= nn; ++i) {
                fill_cand(i, true);
                if (count_terms(cand.as_ref()) == j && divides()) {
                    out.rows_col_major = Array<u8>::from(cand.as_ref(), alloc);
                    out.num_polys = 1;
                    return CommStatus::kOk;
                }
            }
        }
        return CommStatus::kOk;
    }

    if (mode == CyclPolyMode::kMax) {
        for (i32 j = m + 1; j >= 2; --j) {
            for (u64 i = 0; i <= nn; ++i) {
                fill_cand(i, false);
                if (count_terms(cand.as_ref()) == j && divides()) {
                    out.rows_col_major = Array<u8>::from(cand.as_ref(), alloc);
                    out.num_polys = 1;
                    return CommStatus::kOk;
                }
            }
        }
        return CommStatus::kOk;
    }

    // kAll: collect all divisors, stably sorted by Hamming weight ascending.
    struct PolyEntry {
        usize row_idx;
        i32 weight;
    };
    Array<u8> flat_rows(scratch.allocator());
    Array<PolyEntry> entries(scratch.allocator());
    for (u64 i = 0; i <= nn; ++i) {
        fill_cand(i, true);
        if (divides()) {
            const usize ridx = entries.size();
            entries.push_back({ridx, count_terms(cand.as_ref())});
            for (usize c = 0; c < plen; ++c) flat_rows.push_back(cand[c]);
        }
    }
    const usize num_found = entries.size();
    if (num_found == 0) return CommStatus::kOk;

    keel::sort(entries.as_mut_ref(), [](const PolyEntry& a, const PolyEntry& b) {
        if (a.weight != b.weight) return a.weight < b.weight;
        return a.row_idx < b.row_idx;
    });

    out.rows_col_major = Array<u8>(num_found * plen, 0u, alloc);
    for (usize r = 0; r < num_found; ++r) {
        const usize src_off = entries[r].row_idx * plen;
        for (usize c = 0; c < plen; ++c) {
            out.rows_col_major[c * num_found + r] = flat_rows[src_off + c];
        }
    }
    out.num_polys = num_found;
    return CommStatus::kOk;
}

CommStatus cyclgen(i32 n, ArrayRef<u8> gen_poly, bool systematic, BlockCodeMatrices& out,
                   Allocator* alloc) {
    const i32 m = poly_deg(gen_poly);
    if (m < 1 || n <= m) {
        return CommStatus::kInvalidArgument;
    }
    const i32 k = n - m;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> pp(as<usize>(n) + 1u, 0u, scratch.allocator());
    pp[0] = 1u;
    pp[as<usize>(n)] = 1u;
    if (poly_deg(poly_mod(pp.as_ref(), gen_poly, scratch.allocator()).as_ref()) >= 0) {
        return CommStatus::kInvalidArgument;
    }

    const usize nu = as<usize>(n);
    const usize ku = as<usize>(k);
    const usize mu = as<usize>(m);

    if (!systematic) {
        Array<u8> pc = poly_div_mod(pp.as_ref(), gen_poly, scratch.allocator()).q;
        pc.resize(ku + 1u, 0u);
        Array<u8> hrow(nu, 0u, scratch.allocator());
        for (usize j = 0; j <= ku; ++j) hrow[j] = pc[ku - j] & 1u;
        out.h = shift_matrix_col_major(hrow.as_ref(), mu, nu, alloc);

        Array<u8> grow(nu, 0u, scratch.allocator());
        for (usize j = 0; j < gen_poly.size() && j < nu; ++j) grow[j] = gen_poly[j] & 1u;
        out.g = shift_matrix_col_major(grow.as_ref(), ku, nu, alloc);
    } else {
        Array<u8> b_flat(ku * mu, 0u, scratch.allocator());
        for (usize i = 0; i < ku; ++i) {
            Array<u8> xpow(mu + i + 1u, 0u, scratch.allocator());
            xpow[mu + i] = 1u;
            Array<u8> rem = poly_mod(xpow.as_ref(), gen_poly, scratch.allocator());
            for (usize c = 0; c < rem.size() && c < mu; ++c) {
                b_flat[i * mu + c] = rem[c] & 1u;
            }
        }
        out.h = Array<u8>(mu * nu, 0u, alloc);
        for (usize r = 0; r < mu; ++r) {
            for (usize c = 0; c < mu; ++c) {
                out.h[c * mu + r] = (r == c) ? 1u : 0u;
            }
            for (usize t = 0; t < ku; ++t) {
                out.h[(mu + t) * mu + r] = b_flat[t * mu + r];
            }
        }
        out.g = Array<u8>(ku * nu, 0u, alloc);
        for (usize r = 0; r < ku; ++r) {
            for (usize c = 0; c < mu; ++c) {
                out.g[c * ku + r] = b_flat[r * mu + c];
            }
            for (usize t = 0; t < ku; ++t) {
                out.g[(mu + t) * ku + r] = (r == t) ? 1u : 0u;
            }
        }
    }
    out.n = nu;
    out.k = ku;
    return CommStatus::kOk;
}

CommStatus syndtable(ArrayRef<u8> h_col_major, usize nk, usize n, MutArrayRef<u8> out_col_major,
                     Allocator* alloc) {
    if (nk == 0 || n == 0 || nk > 24 || h_col_major.size() / n < nk) {
        return CommStatus::kInvalidArgument;
    }
    const usize num_synd = usize{1} << nk;
    if (out_col_major.size() / n < num_synd) {
        return CommStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<u32> col_synd(n, 0u, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        u32 s = 0u;
        for (usize i = 0; i < nk; ++i) {
            if (col_bit(h_col_major, nk, i, j) != 0u) {
                s |= (u32{1} << (nk - 1u - i));
            }
        }
        col_synd[j] = s;
    }

    for (usize i = 0; i < num_synd * n; ++i) out_col_major[i] = 0u;

    Array<u8> filled(num_synd, 0u, scratch.allocator());
    filled[0] = 1u;
    usize cnt = 1u;

    Array<usize> comb(scratch.allocator());
    for (usize w = 1; w <= n && cnt < num_synd; ++w) {
        comb.resize(w);
        for (usize i = 0; i < w; ++i) comb[i] = i;
        do {
            u32 s = 0u;
            for (usize i = 0; i < w; ++i) s ^= col_synd[comb[i]];
            if (filled[s] == 0u) {
                filled[s] = 1u;
                for (usize i = 0; i < w; ++i) {
                    out_col_major[comb[i] * num_synd + as<usize>(s)] = 1u;
                }
                if (++cnt == num_synd) break;
            }
        } while (next_combination(comb.as_mut_ref(), n));
    }
    return CommStatus::kOk;
}

CommStatus block_encode(ArrayRef<u8> msg_words_row_major, usize num_words, ArrayRef<u8> g_col_major,
                        usize k, usize n, MutArrayRef<u8> code_words_row_major) noexcept {
    if (k == 0 || n <= k || g_col_major.size() / n < k ||
        msg_words_row_major.size() / k < num_words ||
        code_words_row_major.size() / n < num_words) {
        return CommStatus::kInvalidArgument;
    }
    for (usize w = 0; w < num_words; ++w) {
        const usize m_off = w * k;
        const usize c_off = w * n;
        for (usize c = 0; c < n; ++c) {
            u8 b = 0u;
            for (usize p = 0; p < k; ++p) {
                b ^= ((msg_words_row_major[m_off + p] & 1u) & col_bit(g_col_major, k, p, c));
            }
            code_words_row_major[c_off + c] = b & 1u;
        }
    }
    return CommStatus::kOk;
}

CommStatus block_decode(ArrayRef<u8> code_words_row_major, usize num_words,
                        ArrayRef<u8> g_col_major, ArrayRef<u8> h_col_major, usize k, usize n,
                        MutArrayRef<u8> msg_words_row_major, MutArrayRef<i32> err_counts,
                        Allocator* alloc) {
    if (k == 0 || n <= k || g_col_major.size() / n < k || h_col_major.size() / n < (n - k) ||
        code_words_row_major.size() / n < num_words ||
        msg_words_row_major.size() / k < num_words || err_counts.size() < num_words) {
        return CommStatus::kInvalidArgument;
    }
    const usize nk = n - k;
    if (nk > 24) {
        return CommStatus::kInvalidArgument;
    }
    const usize num_synd = usize{1} << nk;
    if (num_synd > Limits<usize>::kMax / n) {
        return CommStatus::kInvalidArgument;
    }

    bool first_k = true;
    bool last_k = true;
    for (usize i = 0; i < k && (first_k || last_k); ++i) {
        for (usize j = 0; j < k; ++j) {
            const u8 want = (i == j) ? 1u : 0u;
            if (col_bit(g_col_major, k, i, j) != want) first_k = false;
            if (col_bit(g_col_major, k, i, nk + j) != want) last_k = false;
        }
    }
    if (!first_k && !last_k) {
        return CommStatus::kNotSystematic;
    }
    const usize msg_off = first_k ? 0u : nk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> table(num_synd * n, 0u, scratch.allocator());
    const CommStatus st = syndtable(h_col_major, nk, n, table.as_mut_ref(), scratch.allocator());
    if (st != CommStatus::kOk) return st;

    for (usize w = 0; w < num_words; ++w) {
        const usize c_off = w * n;
        const usize m_off = w * k;
        usize synd = 0u;
        for (usize j = 0; j < nk; ++j) {
            u8 b = 0u;
            for (usize i = 0; i < n; ++i) {
                b ^= (col_bit(h_col_major, nk, j, i) & (code_words_row_major[c_off + i] & 1u));
            }
            synd = (synd << 1) | as<usize>(b & 1u);
        }
        for (usize p = 0; p < k; ++p) {
            const u8 leader_bit = table[(msg_off + p) * num_synd + synd];
            msg_words_row_major[m_off + p] = (code_words_row_major[c_off + msg_off + p] & 1u) ^ leader_bit;
        }
        i32 cnt = 0;
        for (usize p = 0; p < n; ++p) {
            cnt += as<i32>(table[p * num_synd + synd]);
        }
        err_counts[w] = cnt;
    }
    return CommStatus::kOk;
}

} // namespace keel::comm
