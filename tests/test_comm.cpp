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

} // namespace

int main() {
    test_modulation();
    test_pulse_and_compand();
    test_scrambler_and_errors();

    if (g_failures == 0) {
        std::printf("ALL keel::comm TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
