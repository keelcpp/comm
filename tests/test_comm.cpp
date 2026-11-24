#include "keel/comm.h"

#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace cm = keel::comm;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        std::fprintf(stderr, "FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.3g)\n",
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

} // namespace

int main() {
    test_modulation();
    test_pulse_and_compand();
    test_scrambler_and_errors();
    test_block_coding();
    test_conv_coding();
    test_source_coding();
    test_fsk_ofdm_apsk_analog();
    test_mil188_and_channel();

    if (g_failures == 0) {
        std::printf("ALL keel::comm TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
