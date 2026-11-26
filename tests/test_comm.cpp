#include "keel/comm.h"

#include "keel/format.h"
#include "keel/math.h"

namespace {

using namespace keel;
namespace cm = keel::comm;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        keel::eprintln("FAIL line %d: %s", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        keel::eprintln("FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.15g)",
                       line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

void test_modulation() {
    // Gray roundtrip
    for (u32 i = 0; i < 64; ++i) {
        CHECK(cm::from_gray(cm::to_gray(i)) == i);
    }

    // 8-PSK roundtrip
    const i32 syms8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    c64 mod8[8]{};
    i32 dem8[8]{};
    CHECK(cm::psk_mod(syms8, 8, kPi * 0.125, cm::SymbolOrder::kGray, mod8) == cm::CommStatus::kOk);
    CHECK(cm::psk_demod(mod8, 8, kPi * 0.125, cm::SymbolOrder::kGray, dem8) == cm::CommStatus::kOk);
    for (usize i = 0; i < 8; ++i) CHECK(dem8[i] == syms8[i]);

    // 4-DPSK roundtrip
    const i32 syms4[6] = {0, 3, 2, 1, 3, 0};
    c64 dmod[6]{};
    i32 ddem[6]{};
    CHECK(cm::dpsk_mod(syms4, 4, 0.2, cm::SymbolOrder::kGray, dmod) == cm::CommStatus::kOk);
    CHECK(cm::dpsk_demod(dmod, 4, 0.2, cm::SymbolOrder::kGray, ddem) == cm::CommStatus::kOk);
    for (usize i = 0; i < 6; ++i) CHECK(ddem[i] == syms4[i]);

    // 4-PAM roundtrip
    CHECK(cm::pam_mod(syms4, 4, 0.3, cm::SymbolOrder::kGray, dmod) == cm::CommStatus::kOk);
    CHECK(cm::pam_demod(dmod, 4, 0.3, cm::SymbolOrder::kGray, ddem) == cm::CommStatus::kOk);
    for (usize i = 0; i < 6; ++i) CHECK(ddem[i] == syms4[i]);

    // 16-QAM roundtrip + unit average power check
    i32 syms16[16]{};
    c64 qmod[16]{};
    i32 qdem[16]{};
    for (i32 i = 0; i < 16; ++i) syms16[keel::as<usize>(i)] = i;
    CHECK(cm::qam_mod(syms16, 16, cm::SymbolOrder::kGray, true, qmod) == cm::CommStatus::kOk);
    CHECK(cm::qam_demod(qmod, 16, cm::SymbolOrder::kGray, true, qdem) == cm::CommStatus::kOk);
    f64 avg_p = 0.0;
    for (usize i = 0; i < 16; ++i) {
        CHECK(qdem[i] == syms16[i]);
        avg_p += keel::norm(qmod[i]);
    }
    CHECK_NEAR(avg_p / 16.0, 1.0, 1e-13);

    // modnorm
    CHECK_NEAR(cm::modnorm(qmod, cm::NormMode::kAveragePower, 1.0), 1.0, 1e-13);
}

void test_pulse_and_compand() {
    // RRC unit energy & symmetry
    f64 h[49]{};
    CHECK(cm::rcos_design(0.25, 6, 8, cm::RcosShape::kSqrt, h) == cm::CommStatus::kOk);
    f64 e = 0.0;
    for (usize i = 0; i < 49; ++i) e += h[i] * h[i];
    CHECK_NEAR(e, 1.0, 1e-13);
    for (usize i = 0; i < 24; ++i) CHECK_NEAR(h[i], h[48 - i], 1e-14);

    // RC normal unit energy
    CHECK(cm::rcos_design(0.5, 6, 8, cm::RcosShape::kNormal, h) == cm::CommStatus::kOk);
    e = 0.0;
    for (usize i = 0; i < 49; ++i) e += h[i] * h[i];
    CHECK_NEAR(e, 1.0, 1e-13);

    // Gaussian unit area
    f64 g[25]{};
    CHECK(cm::gauss_design(0.3, 3, 8, g) == cm::CommStatus::kOk);
    f64 s = 0.0;
    for (usize i = 0; i < 25; ++i) s += g[i];
    CHECK_NEAR(s, 1.0, 1e-13);

    // Mu-law and A-law compand roundtrip
    const f64 x[5] = {-1.0, -0.3, 0.0, 0.25, 0.9};
    f64 comp[5]{}, rec[5]{};
    CHECK(cm::compand(x, 255.0, 1.0, cm::CompandMode::kMuCompress, comp) == cm::CommStatus::kOk);
    CHECK(cm::compand(comp, 255.0, 1.0, cm::CompandMode::kMuExpand, rec) == cm::CommStatus::kOk);
    for (usize i = 0; i < 5; ++i) CHECK_NEAR(rec[i], x[i], 1e-13);

    CHECK(cm::compand(x, 87.6, 1.0, cm::CompandMode::kACompress, comp) == cm::CommStatus::kOk);
    CHECK(cm::compand(comp, 87.6, 1.0, cm::CompandMode::kAExpand, rec) == cm::CommStatus::kOk);
    for (usize i = 0; i < 5; ++i) CHECK_NEAR(rec[i], x[i], 1e-13);
}

void test_scrambler_and_errors() {
    const u8 poly[5] = {1, 0, 1, 0, 1}; // 1 + z^-2 + z^-4
    const u8 in_bits[12] = {1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0};
    u8 st1[4] = {1, 0, 1, 0};
    u8 st2[4] = {1, 0, 1, 0};
    u8 scr[12]{};
    u8 dsc[12]{};
    CHECK(cm::scramble(in_bits, poly, st1, scr) == cm::CommStatus::kOk);
    CHECK(cm::descramble(scr, poly, st2, dsc) == cm::CommStatus::kOk);
    for (usize i = 0; i < 12; ++i) CHECK(dsc[i] == in_bits[i]);

    // BER & SER
    const u64 a[4] = {0b000, 0b101, 0b111, 0b010};
    const u64 b[4] = {0b001, 0b101, 0b011, 0b010};
    const cm::ErrorStats ber = cm::bit_error_rate(a, b, 3);
    CHECK(ber.count == 2);
    CHECK_NEAR(ber.ratio, 2.0 / 12.0, 1e-15);

    const f64 fa[4] = {1.0, 2.0, 3.0, 4.0};
    const f64 fb[4] = {1.0, 9.0, 3.0, 4.0};
    const cm::ErrorStats ser = cm::sym_error_rate(fa, fb);
    CHECK(ser.count == 1);
    CHECK_NEAR(ser.ratio, 0.25, 1e-15);
}

void test_block_coding() {
    // Hamming(7, 4): single-bit error correction
    cm::BlockCodeMatrices ham;
    CHECK(cm::hammgen(3, 0, ham) == cm::CommStatus::kOk);
    CHECK(ham.n == 7 && ham.k == 4);

    // gen2par involution
    keel::Array<u8> h_back;
    CHECK(cm::gen2par(ham.g.as_ref(), 4, 7, h_back) == cm::CommStatus::kOk);
    CHECK(h_back.size() == ham.h.size());
    for (usize i = 0; i < ham.h.size(); ++i) CHECK(h_back[i] == ham.h[i]);

    const u8 msg[8] = {1, 0, 1, 1, 0, 1, 0, 1};
    u8 code[14]{};
    CHECK(cm::block_encode(msg, 2, ham.g.as_ref(), 4, 7, code) == cm::CommStatus::kOk);

    // Flip 1 bit in each codeword
    code[2] ^= 1u;
    code[7 + 5] ^= 1u;
    u8 dec[8]{};
    i32 errs[2]{};
    CHECK(cm::block_decode(code, 2, ham.g.as_ref(), ham.h.as_ref(), 4, 7, dec, errs) ==
          cm::CommStatus::kOk);
    CHECK(errs[0] == 1 && errs[1] == 1);
    for (usize i = 0; i < 8; ++i) CHECK(dec[i] == msg[i]);

    // cyclpoly + cyclgen(7, 4)
    cm::CyclPolyResult cp;
    CHECK(cm::cyclpoly(7, 4, cm::CyclPolyMode::kAll, cp) == cm::CommStatus::kOk);
    CHECK(cp.num_polys == 2 && cp.poly_len == 4);

    const u8 gpoly[4] = {1, 1, 0, 1};
    cm::BlockCodeMatrices cyc;
    CHECK(cm::cyclgen(7, gpoly, true, cyc) == cm::CommStatus::kOk);
    CHECK(cyc.n == 7 && cyc.k == 4);

    // syndtable(H) has 8 rows x 7 cols
    u8 tbl[8 * 7]{};
    CHECK(cm::syndtable(ham.h.as_ref(), 3, 7, tbl) == cm::CommStatus::kOk);
    for (usize c = 0; c < 7; ++c) CHECK(tbl[c * 8 + 0] == 0u);
}

void test_conv_coding() {
    const u64 gens[2] = {6, 7}; // octal [6 7], K = 3
    cm::Trellis tr;
    CHECK(cm::poly2trellis(3, gens, tr) == cm::CommStatus::kOk);
    CHECK(tr.num_input_symbols == 2 && tr.num_output_symbols == 4 && tr.num_states == 4);

    // Terminated message (with 2 zero tail bits)
    const u8 msg[10] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 0};
    u8 enc[20]{};
    CHECK(cm::convenc(msg, tr.next_states.as_ref(), tr.outputs.as_ref(), tr.num_states, 2, enc) ==
          cm::CommStatus::kOk);

    // Flip 1 channel bit
    enc[5] ^= 1u;
    u8 dec[10]{};
    CHECK(cm::vitdec_hard(enc, tr.next_states.as_ref(), tr.outputs.as_ref(), tr.num_states, 2, true,
                          dec) == cm::CommStatus::kOk);
    for (usize i = 0; i < 10; ++i) CHECK(dec[i] == msg[i]);
}

void test_source_coding() {
    // Huffman roundtrip
    const f64 probs[4] = {0.5, 0.25, 0.125, 0.125};
    cm::HuffmanDict dict;
    CHECK(cm::huffman_dict(probs, dict) == cm::CommStatus::kOk);
    CHECK_NEAR(dict.avg_len, 1.75, 1e-12);

    const usize seq[6] = {0, 1, 2, 3, 0, 1};
    keel::Array<u8> hbits;
    CHECK(cm::huffman_encode(seq, dict.offsets.as_ref(), dict.bits.as_ref(), hbits) ==
          cm::CommStatus::kOk);
    keel::Array<usize> hdec;
    CHECK(cm::huffman_decode(hbits.as_ref(), dict.offsets.as_ref(), dict.bits.as_ref(), hdec) ==
          cm::CommStatus::kOk);
    CHECK(hdec.size() == 6);
    for (usize i = 0; i < 6; ++i) CHECK(hdec[i] == seq[i]);

    // Arithmetic coding roundtrip
    const u64 counts[3] = {50, 30, 20};
    const usize aseq[8] = {1, 2, 1, 3, 2, 1, 1, 3};
    keel::Array<u8> abits;
    CHECK(cm::arith_encode(aseq, counts, abits) == cm::CommStatus::kOk);
    usize adec[8]{};
    CHECK(cm::arith_decode(abits.as_ref(), counts, 8, adec) == cm::CommStatus::kOk);
    for (usize i = 0; i < 8; ++i) CHECK(adec[i] == aseq[i]);

    // Quantiz + Lloyds + DPCM + dpcm_opt
    const f64 sig[5] = {-2.0, -0.5, 0.1, 0.9, 2.5};
    const f64 part[3] = {-1.0, 0.0, 1.0};
    const f64 cb[4] = {-1.5, -0.5, 0.5, 1.5};
    usize qidx[5]{};
    f64 qval[5]{};
    f64 distor = 0.0;
    CHECK(cm::quantiz(sig, part, cb, qidx, qval, distor) == cm::CommStatus::kOk);
    CHECK(qidx[0] == 0 && qidx[1] == 1 && qidx[2] == 2 && qidx[4] == 3);

    f64 train[64]{};
    for (usize i = 0; i < 64; ++i) {
        train[i] = keel::sin(0.2 * keel::as<f64>(i));
    }
    cm::LloydsResult ll;
    CHECK(cm::lloyds(train, {}, 4, 1e-7, ll) == cm::CommStatus::kOk);
    CHECK(ll.codebook.size() == 4 && ll.partition.size() == 3 && ll.distortion < 0.05);

    cm::DpcmOptResult dopt;
    CHECK(cm::dpcm_opt(train, 2, {}, 4, dopt) == cm::CommStatus::kOk);
    CHECK(dopt.predictor.size() == 3 && dopt.predictor[0] == 0.0 && dopt.codebook.size() == 4);

    usize didx[64]{};
    f64 dqerr[64]{}, drec[64]{}, dqerr2[64]{};
    CHECK(cm::dpcm_encode(train, dopt.codebook.as_ref(), dopt.partition.as_ref(),
                          dopt.predictor.as_ref(), didx, dqerr) == cm::CommStatus::kOk);
    CHECK(cm::dpcm_decode(didx, dopt.codebook.as_ref(), dopt.predictor.as_ref(), drec, dqerr2) ==
          cm::CommStatus::kOk);
    for (usize i = 0; i < 64; ++i) CHECK_NEAR(dqerr[i], dqerr2[i], 1e-12);
}

void test_fsk_ofdm_apsk_analog() {
    // 4-FSK roundtrip
    const i32 fsk_in[4] = {0, 1, 2, 3};
    c64 fsk_sig[4 * 16]{};
    i32 fsk_out[4]{};
    CHECK(cm::fsk_mod(fsk_in, 4, 100.0, 16, 1600.0, true, cm::SymbolOrder::kGray, fsk_sig) ==
          cm::CommStatus::kOk);
    CHECK(cm::fsk_demod(fsk_sig, 4, 100.0, 16, 1600.0, cm::SymbolOrder::kGray, fsk_out) ==
          cm::CommStatus::kOk);
    for (usize i = 0; i < 4; ++i) CHECK(fsk_out[i] == fsk_in[i]);

    // OFDM roundtrip (NFFT=16, CP=4, 2 symbols)
    c64 ofdm_in[32]{};
    for (usize i = 0; i < 32; ++i) {
        ofdm_in[i] = c64(keel::as<f64>(i % 4) - 1.5, keel::as<f64>((i / 4) % 4) - 1.5);
    }
    c64 ofdm_tx[40]{}, ofdm_rx[32]{};
    CHECK(cm::ofdm_mod(ofdm_in, 16, 2, 4, ofdm_tx) == cm::CommStatus::kOk);
    CHECK(cm::ofdm_demod(ofdm_tx, 16, 2, 4, 4, ofdm_rx) == cm::CommStatus::kOk);
    for (usize i = 0; i < 32; ++i) {
        CHECK_NEAR(ofdm_rx[i].real(), ofdm_in[i].real(), 1e-12);
        CHECK_NEAR(ofdm_rx[i].imag(), ofdm_in[i].imag(), 1e-12);
    }

    // APSK constellation + genqam roundtrip
    const usize rings[2] = {4, 12};
    const f64 radii[2] = {1.0, 2.5};
    keel::Array<c64> apsk_c;
    CHECK(cm::apsk_constellation(rings, radii, {}, apsk_c) == cm::CommStatus::kOk);
    CHECK(apsk_c.size() == 16);
    const i32 g_in[4] = {0, 3, 7, 15};
    c64 g_mod[4]{};
    i32 g_out[4]{};
    CHECK(cm::genqam_mod(g_in, apsk_c.as_ref(), g_mod) == cm::CommStatus::kOk);
    CHECK(cm::genqam_demod(g_mod, apsk_c.as_ref(), g_out) == cm::CommStatus::kOk);
    for (usize i = 0; i < 4; ++i) CHECK(g_out[i] == g_in[i]);

    // MSK differential mod/demod roundtrip
    const u8 msk_in[8] = {1, 0, 1, 1, 0, 0, 1, 0};
    c64 msk_sig[8 * 8]{};
    u8 msk_out[8]{};
    f64 final_ph = 0.0;
    CHECK(cm::msk_mod_diff(msk_in, 8, 0.0, msk_sig) == cm::CommStatus::kOk);
    CHECK(cm::msk_demod_diff(msk_sig, 8, 0.0, msk_out, &final_ph) == cm::CommStatus::kOk);
    for (usize i = 0; i < 8; ++i) CHECK(msk_out[i] == msk_in[i]);

    // PM mod + demod roundtrip (interior samples away from Hilbert edges)
    f64 msg_pm[128]{}, y_pm[128]{}, rec_pm[128]{};
    for (usize i = 0; i < 128; ++i) {
        msg_pm[i] = 0.5 * keel::sin(keel::kTau * 5.0 * keel::as<f64>(i) / 1000.0);
    }
    CHECK(cm::pm_mod(msg_pm, 200.0, 1000.0, 1.0, 0.0, y_pm) == cm::CommStatus::kOk);
    CHECK(cm::pm_demod(y_pm, 200.0, 1000.0, 1.0, 0.0, rec_pm) == cm::CommStatus::kOk);
    for (usize i = 24; i < 104; ++i) CHECK_NEAR(rec_pm[i], msg_pm[i], 0.05);
}

void test_mil188_and_channel() {
    const i32 orders[4] = {16, 32, 64, 256};
    for (usize oi = 0; oi < 4; ++oi) {
        const i32 m = orders[oi];
        CHECK(cm::mil188_constellation(m).size() == keel::as<usize>(m));
        keel::Array<i32> syms(keel::as<usize>(m), 0);
        for (i32 k = 0; k < m; ++k) syms[keel::as<usize>(k)] = k;
        keel::Array<c64> mod(keel::as<usize>(m), c64(0.0, 0.0));
        keel::Array<i32> rec(keel::as<usize>(m), -1);
        CHECK(cm::mil188_qam_mod(syms.as_ref(), m, mod.as_mut_ref()) == cm::CommStatus::kOk);
        CHECK(cm::mil188_qam_demod(mod.as_ref(), m, rec.as_mut_ref()) == cm::CommStatus::kOk);
        for (usize k = 0; k < keel::as<usize>(m); ++k) CHECK(rec[k] == syms[k]);
    }
    CHECK_NEAR(cm::qfunc(0.0), 0.5, 1e-12);
    CHECK(cm::ber_awgn(10.0, cm::AwgnMod::kPsk, 2) < 1e-5);
    CHECK(cm::ber_awgn(10.0, cm::AwgnMod::kQam, 16) < 0.01);
    const f64 b[1] = {1.0};
    const f64 a[1] = {1.0};
    CHECK_NEAR(cm::noise_bw(b, a, 256, 1000.0), 1000.0, 1e-9);
}

void test_hardening_regressions() {
    const f64 nan_v = Limits<f64>::quiet_nan();
    const f64 inf_v = Limits<f64>::infinity();

    // 1. rcos_design beta == 0 (RRC and normal) and tiny beta: must be finite, unit energy, exact sinc
    {
        f64 h0[49]{}, h_tiny[49]{}, h_norm[49]{};
        CHECK(cm::rcos_design(0.0, 6, 8, cm::RcosShape::kSqrt, h0) == cm::CommStatus::kOk);
        CHECK(cm::rcos_design(1e-15, 6, 8, cm::RcosShape::kSqrt, h_tiny) == cm::CommStatus::kOk);
        CHECK(cm::rcos_design(0.0, 6, 8, cm::RcosShape::kNormal, h_norm) == cm::CommStatus::kOk);
        f64 e0 = 0.0;
        for (usize i = 0; i < 49; ++i) {
            CHECK(keel::is_finite(h0[i]));
            CHECK_NEAR(h0[i], h_norm[i], 1e-14);
            CHECK_NEAR(h0[i], h_tiny[i], 1e-12);
            e0 += h0[i] * h0[i];
        }
        CHECK_NEAR(e0, 1.0, 1e-13);
        CHECK(cm::rcos_design(nan_v, 6, 8, cm::RcosShape::kSqrt, h0) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::rcos_design(1.5, 6, 8, cm::RcosShape::kSqrt, h0) == cm::CommStatus::kInvalidArgument);
    }

    // 2. gauss_design extreme bt (no overflow) and invalid inputs
    {
        f64 g[25]{};
        CHECK(cm::gauss_design(1e100, 3, 8, g) == cm::CommStatus::kOk);
        f64 sum = 0.0;
        for (usize i = 0; i < 25; ++i) {
            CHECK(keel::is_finite(g[i]));
            sum += g[i];
        }
        CHECK_NEAR(sum, 1.0, 1e-13);
        CHECK(cm::gauss_design(nan_v, 3, 8, g) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::gauss_design(inf_v, 3, 8, g) == cm::CommStatus::kInvalidArgument);
    }

    // 3. compand validation + near-zero accuracy
    {
        const f64 tiny[2] = {-1e-12, 1e-12};
        f64 c[2]{}, r[2]{};
        CHECK(cm::compand(tiny, 255.0, 1.0, cm::CompandMode::kMuCompress, c) == cm::CommStatus::kOk);
        CHECK(cm::compand(c, 255.0, 1.0, cm::CompandMode::kMuExpand, r) == cm::CommStatus::kOk);
        CHECK_NEAR(r[0], tiny[0], 1e-24);
        CHECK_NEAR(r[1], tiny[1], 1e-24);
        CHECK(cm::compand(tiny, 0.5, 1.0, cm::CompandMode::kACompress, c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::compand(tiny, 0.5, 1.0, cm::CompandMode::kAExpand, c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::compand(tiny, nan_v, 1.0, cm::CompandMode::kMuCompress, c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::compand(tiny, 255.0, inf_v, cm::CompandMode::kMuCompress, c) == cm::CommStatus::kInvalidArgument);
    }

    // 4. Modulation domain & overflow/UB guards
    {
        const i32 bad_sym[2] = {0, 8};
        const i32 neg_sym[2] = {-1, 0};
        c64 out_c[2]{};
        i32 out_i[2]{};
        CHECK(cm::psk_mod(bad_sym, 8, 0.0, cm::SymbolOrder::kGray, out_c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::psk_mod(neg_sym, 8, 0.0, cm::SymbolOrder::kGray, out_c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::dpsk_mod(bad_sym, 8, 0.0, cm::SymbolOrder::kGray, out_c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::pam_mod(bad_sym, 8, 0.0, cm::SymbolOrder::kGray, out_c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::qam_mod(bad_sym, 4, cm::SymbolOrder::kGray, true, out_c) == cm::CommStatus::kInvalidArgument);
        CHECK(cm::psk_mod(bad_sym, 16, nan_v, cm::SymbolOrder::kGray, out_c) == cm::CommStatus::kInvalidArgument);

        const c64 extreme_c[2] = {c64(1e300, -1e300), c64(nan_v, inf_v)};
        CHECK(cm::psk_demod(extreme_c, 8, 1e12, cm::SymbolOrder::kGray, out_i) == cm::CommStatus::kOk);
        CHECK(out_i[0] >= 0 && out_i[0] < 8 && out_i[1] >= 0 && out_i[1] < 8);
        CHECK(cm::pam_demod(extreme_c, 4, 0.0, cm::SymbolOrder::kGray, out_i) == cm::CommStatus::kOk);
        CHECK(out_i[0] >= 0 && out_i[0] < 4 && out_i[1] >= 0 && out_i[1] < 4);
        CHECK(cm::qam_demod(extreme_c, 16, cm::SymbolOrder::kGray, true, out_i) == cm::CommStatus::kOk);
        CHECK(out_i[0] >= 0 && out_i[0] < 16 && out_i[1] >= 0 && out_i[1] < 16);

        const c64 big_c[2] = {c64(1e200, 0.0), c64(0.0, -1e200)};
        CHECK_NEAR(cm::modnorm(big_c, cm::NormMode::kAveragePower, 1.0), 1e-200, 1e-212);
        CHECK_NEAR(cm::modnorm(big_c, cm::NormMode::kPeakPower, 4.0), 2e-200, 1e-212);
        CHECK(keel::is_nan(cm::modnorm(big_c, cm::NormMode::kAveragePower, -1.0)));
        CHECK(keel::is_nan(cm::modnorm(big_c, cm::NormMode::kAveragePower, nan_v)));
    }

    // 5. Block & conv coding bounds
    {
        cm::CyclPolyResult cp;
        CHECK(cm::cyclpoly(100, 1, cm::CyclPolyMode::kFirst, cp) == cm::CommStatus::kInvalidArgument);
        cm::BlockCodeMatrices bm;
        CHECK(cm::hammgen(3, 0x1BULL, bm) == cm::CommStatus::kInvalidArgument);
        const u64 huge_oct[1] = {7777777777777777777ULL};
        cm::Trellis tr;
        CHECK(cm::poly2trellis(3, huge_oct, tr) == cm::CommStatus::kInvalidArgument);
    }

    // 6. Source coding overflow & malformed dict guards
    {
        const u64 huge_cnt[2] = {1ULL << 31, 1ULL << 31};
        const usize seq1[1] = {1u};
        keel::Array<u8> bits;
        CHECK(cm::arith_encode(seq1, huge_cnt, bits) == cm::CommStatus::kInvalidArgument);
        usize dec0[1]{};
        const u64 ok_cnt[2] = {10, 20};
        CHECK(cm::arith_decode({}, ok_cnt, 0u, dec0) == cm::CommStatus::kOk);

        const u32 bad_off[3] = {0u, 0u, 1u};
        const u8 d_bits[1] = {1u};
        keel::Array<usize> hdec;
        CHECK(cm::huffman_decode(d_bits, bad_off, d_bits, hdec) == cm::CommStatus::kInvalidArgument);

        const f64 bad_train[3] = {0.0, nan_v, 1.0};
        cm::LloydsResult ll;
        CHECK(cm::lloyds(bad_train, {}, 2, 1e-7, ll) == cm::CommStatus::kInvalidArgument);
    }

    // 7. FSK/OFDM/APSK/MSK guards & in-place OFDM
    {
        const i32 bad_fsk[1] = {4};
        c64 fout[8]{};
        CHECK(cm::fsk_mod(bad_fsk, 4, 100.0, 8, 1600.0, true, cm::SymbolOrder::kGray, fout) ==
              cm::CommStatus::kInvalidArgument);
        CHECK(cm::fsk_mod(bad_fsk, 4, nan_v, 8, 1600.0, true, cm::SymbolOrder::kGray, fout) ==
              cm::CommStatus::kInvalidArgument);

        c64 ofdm_ip[16]{};
        c64 ofdm_orig[16]{};
        for (usize i = 0; i < 16; ++i) {
            ofdm_ip[i] = c64(as<f64>(i) - 7.5, as<f64>(i % 3) - 1.0);
            ofdm_orig[i] = ofdm_ip[i];
        }
        CHECK(cm::ofdm_mod(ofdm_ip, 8, 2, 0, ofdm_ip) == cm::CommStatus::kOk);
        CHECK(cm::ofdm_demod(ofdm_ip, 8, 2, 0, 0, ofdm_ip) == cm::CommStatus::kOk);
        for (usize i = 0; i < 16; ++i) {
            CHECK_NEAR(ofdm_ip[i].real(), ofdm_orig[i].real(), 1e-12);
            CHECK_NEAR(ofdm_ip[i].imag(), ofdm_orig[i].imag(), 1e-12);
        }

        const usize rsz[1] = {4};
        const f64 bad_rad[1] = {-1.0};
        keel::Array<c64> apsk_c;
        CHECK(cm::apsk_constellation(rsz, bad_rad, {}, apsk_c) == cm::CommStatus::kInvalidArgument);

        const c64 msk_s[4] = {c64(1.0, 0.0), c64(0.0, 1.0), c64(-1.0, 0.0), c64(0.0, -1.0)};
        u8 msk_b[4]{};
        CHECK(cm::msk_demod_diff(msk_s, 1, 0.0, msk_b) == cm::CommStatus::kInvalidArgument);
    }

    // 8. ber_awgn & noise_bw exactness and edge cases
    {
        CHECK(keel::is_nan(cm::ber_awgn(5.0, cm::AwgnMod::kPsk, 1)));
        CHECK(keel::is_nan(cm::ber_awgn(5.0, cm::AwgnMod::kQam, 2)));
        CHECK(keel::is_nan(cm::ber_awgn(nan_v, cm::AwgnMod::kPsk, 2)));
        CHECK_NEAR(cm::ber_awgn(inf_v, cm::AwgnMod::kPsk, 2), 0.0, 1e-15);
        CHECK_NEAR(cm::ber_awgn(-inf_v, cm::AwgnMod::kPsk, 2), 0.5, 1e-15);

        const f64 b[3] = {0.2, 0.4, 0.2};
        const f64 a[3] = {1.0, -0.5, 0.25};
        CHECK_NEAR(cm::noise_bw(b, a, 256, 1000.0), 406.25, 1e-11);
        CHECK_NEAR(cm::noise_bw(b, {}, 256, 1000.0), 0.0, 1e-15);
        CHECK(keel::is_nan(cm::noise_bw(b, a, 256, -1.0)));
        CHECK(keel::is_nan(cm::noise_bw(b, a, 256, nan_v)));
    }
}

void print_f64_vec(const char* key, ArrayRef<f64> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%.17g", v[i]);
    }
    keel::print("\n");
}

void print_c64_vec(const char* key, ArrayRef<c64> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%.17g,%.17g", v[i].real(), v[i].imag());
    }
    keel::print("\n");
}

void print_u8_vec(const char* key, ArrayRef<u8> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%u", as<u32>(v[i]));
    }
    keel::print("\n");
}

void print_u32_vec(const char* key, ArrayRef<u32> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%u", v[i]);
    }
    keel::print("\n");
}

void print_usize_vec(const char* key, ArrayRef<usize> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%llu", as<u64>(v[i]));
    }
    keel::print("\n");
}

void dump_reference() {
    // 1. PSK / DPSK / PAM / QAM / modnorm
    const i32 s8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    c64 psk8_g[8]{}, psk8_b[8]{};
    (void)cm::psk_mod(s8, 8, kPi * 0.125, cm::SymbolOrder::kGray, psk8_g);
    (void)cm::psk_mod(s8, 8, 0.0, cm::SymbolOrder::kBinary, psk8_b);
    print_c64_vec("PSK8_GRAY", psk8_g);
    print_c64_vec("PSK8_BIN", psk8_b);

    const i32 s4[6] = {0, 3, 2, 1, 3, 0};
    c64 dpsk4[6]{}, pam4_g[6]{}, pam4_b[6]{};
    (void)cm::dpsk_mod(s4, 4, 0.25, cm::SymbolOrder::kGray, dpsk4);
    (void)cm::pam_mod(s4, 4, 0.3, cm::SymbolOrder::kGray, pam4_g);
    (void)cm::pam_mod(s4, 4, 0.0, cm::SymbolOrder::kBinary, pam4_b);
    print_c64_vec("DPSK4_GRAY", dpsk4);
    print_c64_vec("PAM4_GRAY", pam4_g);
    print_c64_vec("PAM4_BIN", pam4_b);

    i32 s16[16]{};
    for (i32 i = 0; i < 16; ++i) s16[as<usize>(i)] = i;
    c64 qam16_g[16]{}, qam16_u[16]{}, qam16_b[16]{};
    (void)cm::qam_mod(s16, 16, cm::SymbolOrder::kGray, false, qam16_g);
    (void)cm::qam_mod(s16, 16, cm::SymbolOrder::kGray, true, qam16_u);
    (void)cm::qam_mod(s16, 16, cm::SymbolOrder::kBinary, false, qam16_b);
    print_c64_vec("QAM16_GRAY_RAW", qam16_g);
    print_c64_vec("QAM16_GRAY_UNIT", qam16_u);
    print_c64_vec("QAM16_BIN_RAW", qam16_b);

    c64 pam8_g[8]{};
    (void)cm::pam_mod(s8, 8, 0.0, cm::SymbolOrder::kGray, pam8_g);
    print_c64_vec("PAM8_GRAY", pam8_g);

    i32 s64[64]{};
    for (i32 i = 0; i < 64; ++i) s64[as<usize>(i)] = i;
    c64 qam64_g[64]{};
    (void)cm::qam_mod(s64, 64, cm::SymbolOrder::kGray, false, qam64_g);
    print_c64_vec("QAM64_GRAY_RAW", qam64_g);

    c64 fsk8_g[8 * 8]{};
    (void)cm::fsk_mod(s8, 8, 100.0, 8, 16000.0, true, cm::SymbolOrder::kGray, fsk8_g);
    print_c64_vec("FSK8_GRAY_CONT", fsk8_g);

    const f64 mn_vals[2] = {
        cm::modnorm(qam16_g, cm::NormMode::kAveragePower, 1.0),
        cm::modnorm(qam16_g, cm::NormMode::kPeakPower, 2.0),
    };
    print_f64_vec("MODNORM_QAM16", mn_vals);

    // 2. Pulse shaping & companding
    f64 rrc_25[49]{}, rrc_0[49]{}, rc_50[49]{}, g_30[25]{};
    (void)cm::rcos_design(0.25, 6, 8, cm::RcosShape::kSqrt, rrc_25);
    (void)cm::rcos_design(0.0, 6, 8, cm::RcosShape::kSqrt, rrc_0);
    (void)cm::rcos_design(0.5, 6, 8, cm::RcosShape::kNormal, rc_50);
    (void)cm::gauss_design(0.3, 3, 8, g_30);
    print_f64_vec("RCOS_SQRT_025", rrc_25);
    print_f64_vec("RCOS_SQRT_000", rrc_0);
    print_f64_vec("RCOS_NORM_050", rc_50);
    print_f64_vec("GAUSS_030", g_30);

    const f64 cx[6] = {-1.0, -0.4, -0.005, 0.0, 0.25, 0.9};
    f64 mu_c[6]{}, mu_e[6]{}, a_c[6]{}, a_e[6]{};
    (void)cm::compand(cx, 255.0, 1.0, cm::CompandMode::kMuCompress, mu_c);
    (void)cm::compand(mu_c, 255.0, 1.0, cm::CompandMode::kMuExpand, mu_e);
    (void)cm::compand(cx, 87.6, 1.0, cm::CompandMode::kACompress, a_c);
    (void)cm::compand(a_c, 87.6, 1.0, cm::CompandMode::kAExpand, a_e);
    print_f64_vec("COMPAND_MU_C", mu_c);
    print_f64_vec("COMPAND_MU_E", mu_e);
    print_f64_vec("COMPAND_A_C", a_c);
    print_f64_vec("COMPAND_A_E", a_e);

    // 3. Block coding
    cm::BlockCodeMatrices ham;
    (void)cm::hammgen(3, 0, ham);
    print_u8_vec("HAMM3_H", ham.h.as_ref());
    print_u8_vec("HAMM3_G", ham.g.as_ref());

    cm::CyclPolyResult cp_min, cp_all;
    (void)cm::cyclpoly(7, 4, cm::CyclPolyMode::kMin, cp_min);
    (void)cm::cyclpoly(7, 4, cm::CyclPolyMode::kAll, cp_all);
    print_u8_vec("CYCLPOLY_7_4_MIN", cp_min.rows_col_major.as_ref());
    print_u8_vec("CYCLPOLY_7_4_ALL", cp_all.rows_col_major.as_ref());

    const u8 gp[4] = {1, 1, 0, 1};
    cm::BlockCodeMatrices cyc_s, cyc_n;
    (void)cm::cyclgen(7, gp, true, cyc_s);
    (void)cm::cyclgen(7, gp, false, cyc_n);
    print_u8_vec("CYCLGEN_SYS_H", cyc_s.h.as_ref());
    print_u8_vec("CYCLGEN_SYS_G", cyc_s.g.as_ref());
    print_u8_vec("CYCLGEN_NOSYS_G", cyc_n.g.as_ref());

    u8 stbl[8 * 7]{};
    (void)cm::syndtable(ham.h.as_ref(), 3, 7, stbl);
    print_u8_vec("SYNDTABLE_HAMM3", stbl);

    // 4. Convolutional coding
    const u64 gens[2] = {6, 7};
    cm::Trellis tr;
    (void)cm::poly2trellis(3, gens, tr);
    print_u32_vec("TRELLIS_NEXT", tr.next_states.as_ref());
    print_u32_vec("TRELLIS_OUT", tr.outputs.as_ref());
    const u8 cmsg[10] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 0};
    u8 cenc[20]{}, cdec[10]{};
    (void)cm::convenc(cmsg, tr.next_states.as_ref(), tr.outputs.as_ref(), tr.num_states, 2, cenc);
    print_u8_vec("CONVENC_6_7", cenc);
    cenc[3] ^= 1u;
    (void)cm::vitdec_hard(cenc, tr.next_states.as_ref(), tr.outputs.as_ref(), tr.num_states, 2,
                          true, cdec);
    print_u8_vec("VITDEC_6_7", cdec);

    // 5. Source coding (Huffman, Arithmetic, Quantiz, Lloyds, DPCM)
    const f64 hprobs[4] = {0.5, 0.25, 0.125, 0.125};
    cm::HuffmanDict hdict;
    (void)cm::huffman_dict(hprobs, hdict);
    const f64 havglen[1] = {hdict.avg_len};
    print_f64_vec("HUFF_AVGLEN", havglen);

    const u64 acnts[3] = {50, 30, 20};
    const usize aseq[8] = {1, 2, 1, 3, 2, 1, 1, 3};
    keel::Array<u8> abits;
    (void)cm::arith_encode(aseq, acnts, abits);
    print_u8_vec("ARITH_BITS", abits.as_ref());

    const f64 qsig[5] = {-2.0, -0.5, 0.1, 0.9, 2.5};
    const f64 qpart[3] = {-1.0, 0.0, 1.0};
    const f64 qcb[4] = {-1.5, -0.5, 0.5, 1.5};
    usize qidx[5]{};
    f64 qval[5]{};
    f64 qdist = 0.0;
    (void)cm::quantiz(qsig, qpart, qcb, qidx, qval, qdist);
    print_usize_vec("QUANTIZ_IDX", qidx);
    print_f64_vec("QUANTIZ_VAL", qval);
    const f64 qdist_v[1] = {qdist};
    print_f64_vec("QUANTIZ_DIST", qdist_v);

    f64 train[64]{};
    for (usize i = 0; i < 64; ++i) train[i] = keel::sin(0.2 * as<f64>(i));
    cm::LloydsResult ll;
    (void)cm::lloyds(train, {}, 4, 1e-7, ll);
    print_f64_vec("LLOYDS_PART", ll.partition.as_ref());
    print_f64_vec("LLOYDS_CB", ll.codebook.as_ref());

    cm::DpcmOptResult dopt;
    (void)cm::dpcm_opt(train, 2, {}, 4, dopt);
    print_f64_vec("DPCMOPT_PRED", dopt.predictor.as_ref());
    print_f64_vec("DPCMOPT_CB", dopt.codebook.as_ref());

    // 6. FSK, OFDM, APSK, PM/AM/FM/SSB
    const i32 fsk_in[4] = {0, 2, 3, 1};
    c64 fsk_cont[4 * 8]{}, fsk_disc[4 * 8]{};
    (void)cm::fsk_mod(fsk_in, 4, 100.0, 8, 1600.0, true, cm::SymbolOrder::kBinary, fsk_cont);
    (void)cm::fsk_mod(fsk_in, 4, 100.0, 8, 1600.0, false, cm::SymbolOrder::kBinary, fsk_disc);
    print_c64_vec("FSK4_CONT", fsk_cont);
    print_c64_vec("FSK4_DISC", fsk_disc);

    c64 ofdm_in[16]{};
    for (usize i = 0; i < 16; ++i) ofdm_in[i] = c64(as<f64>(i % 4) - 1.5, as<f64>(i / 4) - 1.5);
    c64 ofdm_tx[20]{};
    (void)cm::ofdm_mod(ofdm_in, 8, 2, 2, ofdm_tx);
    print_c64_vec("OFDM_TX", ofdm_tx);

    f64 xsig[32]{}, y_pm[32]{}, y_am[32]{}, y_fm[32]{}, y_ssb_l[32]{}, y_ssb_u[32]{};
    for (usize i = 0; i < 32; ++i) xsig[i] = 0.4 * keel::sin(kTau * 50.0 * as<f64>(i) / 1000.0);
    (void)cm::pm_mod(xsig, 200.0, 1000.0, 0.8, 0.2, y_pm);
    (void)cm::am_mod(xsig, 200.0, 1000.0, 0.2, 1.5, y_am);
    (void)cm::fm_mod(xsig, 200.0, 1000.0, 40.0, 0.2, y_fm);
    (void)cm::ssb_mod(xsig, 200.0, 1000.0, 0.2, false, y_ssb_l);
    (void)cm::ssb_mod(xsig, 200.0, 1000.0, 0.2, true, y_ssb_u);
    print_f64_vec("PM_MOD", y_pm);
    print_f64_vec("AM_MOD", y_am);
    print_f64_vec("FM_MOD", y_fm);
    print_f64_vec("SSB_LOWER", y_ssb_l);
    print_f64_vec("SSB_UPPER", y_ssb_u);

    // 7. qfunc, ber_awgn, noise_bw
    const f64 qf_vals[5] = {
        cm::qfunc(-2.0), cm::qfunc(0.0), cm::qfunc(1.0), cm::qfunc(2.5), cm::qfunc(5.0)
    };
    print_f64_vec("QFUNC", qf_vals);

    const f64 ber_vals[7] = {
        cm::ber_awgn(6.0, cm::AwgnMod::kPsk, 2),
        cm::ber_awgn(6.0, cm::AwgnMod::kPsk, 4),
        cm::ber_awgn(8.0, cm::AwgnMod::kPsk, 8),
        cm::ber_awgn(10.0, cm::AwgnMod::kQam, 16),
        cm::ber_awgn(8.0, cm::AwgnMod::kPam, 4),
        cm::ber_awgn(6.0, cm::AwgnMod::kFsk, 2),
        cm::ber_awgn(6.0, cm::AwgnMod::kDpsk, 2),
    };
    print_f64_vec("BER_AWGN", ber_vals);

    const f64 nb_b1[3] = {0.2, 0.4, 0.2};
    const f64 nb_a1[3] = {1.0, -0.5, 0.25};
    const f64 nb_b2[4] = {0.25, 0.25, 0.25, 0.25};
    const f64 nb_a2[1] = {1.0};
    const f64 nb_vals[2] = {
        cm::noise_bw(nb_b1, nb_a1, 256, 1000.0),
        cm::noise_bw(nb_b2, nb_a2, 256, 800.0),
    };
    print_f64_vec("NOISE_BW", nb_vals);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }

    test_modulation();
    test_pulse_and_compand();
    test_scrambler_and_errors();
    test_block_coding();
    test_conv_coding();
    test_source_coding();
    test_fsk_ofdm_apsk_analog();
    test_mil188_and_channel();
    test_hardening_regressions();

    if (g_failures == 0) {
        keel::println("ALL keel::comm TESTS PASSED");
        return 0;
    }
    keel::eprintln("TOTAL FAILURES: %d", g_failures);
    return 1;
}
