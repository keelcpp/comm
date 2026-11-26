#!/usr/bin/env python3
"""Differential reference verification for keel::comm against NumPy/SciPy and MATLAB R2025b."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import special as sc_special
from scipy import signal as sc_signal


def find_exe() -> Path:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "Release" / "test_comm.exe",
        root / "build" / "test_comm.exe",
        root / "build" / "test_comm",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"test_comm executable not found in {candidates}")


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_f64(s: str) -> np.ndarray:
    if not s:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def parse_c64(s: str) -> np.ndarray:
    raw = parse_f64(s)
    return raw[0::2] + 1j * raw[1::2]


def parse_int(s: str) -> np.ndarray:
    if not s:
        return np.array([], dtype=np.int64)
    return np.array([int(x) for x in s.split(",")], dtype=np.int64)


def run_matlab_ref() -> dict[str, str]:
    script = r"""
format long g;
fmtf = @(name, x) fprintf('%s=%s\n', name, strjoin(arrayfun(@(v) sprintf('%.17g', v), x(:).', 'UniformOutput', false), ','));
fmtc = @(name, z) fmtf(name, reshape([real(z(:).'); imag(z(:).')], 1, []));
fmti = @(name, x) fprintf('%s=%s\n', name, strjoin(arrayfun(@(v) sprintf('%d', round(v)), x(:).', 'UniformOutput', false), ','));

fmtc('PSK8_GRAY', pskmod(0:7, 8, pi/8, 'gray'));
fmtc('PSK8_BIN', pskmod(0:7, 8, 0, 'bin'));
s4 = [0 3 2 1 3 0];
fmtc('DPSK4_GRAY', dpskmod(s4, 4, 0.25, 'gray'));
fmtc('PAM4_GRAY', pammod(s4, 4, 0.3, 'gray'));
fmtc('PAM4_BIN', pammod(s4, 4, 0, 'bin'));
fmtc('QAM16_GRAY_RAW', qammod(0:15, 16, 'gray'));
fmtc('QAM16_GRAY_UNIT', qammod(0:15, 16, 'gray', 'UnitAveragePower', true));
fmtc('QAM16_BIN_RAW', qammod(0:15, 16, 'bin'));
fmtc('PAM8_GRAY', pammod(0:7, 8, 0, 'gray'));
fmtc('QAM64_GRAY_RAW', qammod(0:63, 64, 'gray'));
fmtc('FSK8_GRAY_CONT', fskmod((0:7).', 8, 100, 8, 16000, 'cont', 'gray'));
q16 = qammod(0:15, 16, 'gray');
fmtf('MODNORM_QAM16', [modnorm(q16, 'avpow', 1.0), modnorm(q16, 'peakpow', 2.0)]);

fmtf('RCOS_SQRT_025', rcosdesign(0.25, 6, 8, 'sqrt'));
fmtf('RCOS_SQRT_000', rcosdesign(0.0, 6, 8, 'sqrt'));
fmtf('RCOS_NORM_050', rcosdesign(0.5, 6, 8, 'normal'));
fmtf('GAUSS_030', gaussdesign(0.3, 3, 8));

cx = [-1.0, -0.4, -0.005, 0.0, 0.25, 0.9];
mu_c = compand(cx, 255, 1.0, 'mu/compressor');
mu_e = compand(mu_c, 255, 1.0, 'mu/expander');
a_c = compand(cx, 87.6, 1.0, 'a/compressor');
a_e = compand(a_c, 87.6, 1.0, 'a/expander');
fmtf('COMPAND_MU_C', mu_c);
fmtf('COMPAND_MU_E', mu_e);
fmtf('COMPAND_A_C', a_c);
fmtf('COMPAND_A_E', a_e);

[H, G] = hammgen(3);
fmti('HAMM3_H', H(:));
fmti('HAMM3_G', G(:));
fmti('CYCLPOLY_7_4_MIN', cyclpoly(7, 4, 'min'));
fmti('CYCLPOLY_7_4_ALL', cyclpoly(7, 4, 'all'));
[Hs, Gs] = cyclgen(7, [1 1 0 1], 'system');
[~, Gn] = cyclgen(7, [1 1 0 1], 'nonsys');
fmti('CYCLGEN_SYS_H', Hs(:));
fmti('CYCLGEN_SYS_G', Gs(:));
fmti('CYCLGEN_NOSYS_G', Gn(:));
fmti('SYNDTABLE_HAMM3', syndtable(H));

tr = poly2trellis(3, [6 7]);
fmti('TRELLIS_NEXT', tr.nextStates(:));
fmti('TRELLIS_OUT', oct2dec(tr.outputs(:)));
cmsg = [1 0 1 1 0 1 0 0 0 0];
cenc = convenc(cmsg, tr);
fmti('CONVENC_6_7', cenc);
cenc_err = cenc; cenc_err(4) = bitxor(cenc_err(4), 1);
cdec = vitdec(cenc_err, tr, 5, 'term', 'hard');
fmti('VITDEC_6_7', cdec);

[~, avglen] = huffmandict(1:4, [0.5 0.25 0.125 0.125]);
fmtf('HUFF_AVGLEN', avglen);
abits = arithenco([1 2 1 3 2 1 1 3], [50 30 20]);
fmti('ARITH_BITS', abits);

[qidx, qval, qdist] = quantiz([-2.0 -0.5 0.1 0.9 2.5], [-1.0 0.0 1.0], [-1.5 -0.5 0.5 1.5]);
fmti('QUANTIZ_IDX', qidx);
fmtf('QUANTIZ_VAL', qval);
fmtf('QUANTIZ_DIST', qdist);

train = sin(0.2 * (0:63));
[lp, lcb] = lloyds(train, 4);
fmtf('LLOYDS_PART', lp);
fmtf('LLOYDS_CB', lcb);
[dpred, dcb, ~] = dpcmopt(train, 2, 4);
fmtf('DPCMOPT_PRED', dpred);
fmtf('DPCMOPT_CB', dcb);

fsk_in = [0 2 3 1].';
fmtc('FSK4_CONT', fskmod(fsk_in, 4, 100, 8, 1600, 'cont', 'bin'));
fmtc('FSK4_DISC', fskmod(fsk_in, 4, 100, 8, 1600, 'discont', 'bin'));

xsig = 0.4 * sin(2*pi*50*(0:31)/1000);
fmtf('PM_MOD', pmmod(xsig, 200, 1000, 0.8, 0.2));
fmtf('AM_MOD', ammod(xsig, 200, 1000, 0.2, 1.5));
fmtf('FM_MOD', fmmod(xsig, 200, 1000, 40, 0.2));
fmtf('SSB_LOWER', ssbmod(xsig, 200, 1000, 0.2));
fmtf('SSB_UPPER', ssbmod(xsig, 200, 1000, 0.2, 'upper'));

fmtf('QFUNC', qfunc([-2.0 0.0 1.0 2.5 5.0]));
ber_vals = [ ...
    berawgn(6, 'psk', 2, 'nondiff'), ...
    berawgn(6, 'psk', 4, 'nondiff'), ...
    berawgn(8, 'psk', 8, 'nondiff'), ...
    berawgn(10, 'qam', 16), ...
    berawgn(8, 'pam', 4), ...
    berawgn(6, 'fsk', 2, 'coherent'), ...
    berawgn(6, 'dpsk', 2)];
fmtf('BER_AWGN', ber_vals);

fmtf('NOISE_BW', [ ...
    noisebw([0.2 0.4 0.2], [1 -0.5 0.25], 256, 1000), ...
    noisebw([0.25 0.25 0.25 0.25], 1, 256, 800)]);
fprintf('MATLAB_DONE=1\n');
"""
    proc = subprocess.run(
        ["matlab", "-batch", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    kv = parse_kv(proc.stdout)
    if kv.get("MATLAB_DONE") != "1":
        raise RuntimeError(
            f"MATLAB reference run failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return kv


def main() -> int:
    exe = find_exe()
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    keel_kv = parse_kv(proc.stdout)
    mat_kv = run_matlab_ref()

    passed = 0
    failed = 0

    def check_close(name: str, got: np.ndarray, ref: np.ndarray, atol: float = 1e-12, rtol: float = 1e-12):
        nonlocal passed, failed
        if got.shape != ref.shape:
            print(f"FAIL {name}: shape mismatch got={got.shape} ref={ref.shape}")
            failed += 1
            return
        err = np.max(np.abs(got - ref)) if got.size else 0.0
        if np.allclose(got, ref, atol=atol, rtol=rtol):
            passed += 1
        else:
            print(f"FAIL {name}: max_abs_err={err:.3e} (atol={atol}, rtol={rtol})")
            print(f"  got[:6]={got[:6]}")
            print(f"  ref[:6]={ref[:6]}")
            failed += 1

    def check_exact(name: str, got: np.ndarray, ref: np.ndarray):
        nonlocal passed, failed
        if got.shape == ref.shape and np.array_equal(got, ref):
            passed += 1
        else:
            print(f"FAIL {name}: exact mismatch\n  got={got}\n  ref={ref}")
            failed += 1

    # 1. Digital modulation (vs MATLAB R2025b)
    for k in ["PSK8_GRAY", "PSK8_BIN", "DPSK4_GRAY", "PAM4_GRAY", "PAM4_BIN",
              "QAM16_GRAY_RAW", "QAM16_GRAY_UNIT", "QAM16_BIN_RAW",
              "PAM8_GRAY", "QAM64_GRAY_RAW", "FSK8_GRAY_CONT"]:
        check_close(f"MATLAB:{k}", parse_c64(keel_kv[k]), parse_c64(mat_kv[k]), atol=1e-12, rtol=1e-12)
    check_close("MATLAB:MODNORM_QAM16", parse_f64(keel_kv["MODNORM_QAM16"]), parse_f64(mat_kv["MODNORM_QAM16"]), 1e-13, 1e-13)

    # 2. Pulse shaping & companding (vs MATLAB R2025b)
    for k in ["RCOS_SQRT_025", "RCOS_SQRT_000", "RCOS_NORM_050", "GAUSS_030",
              "COMPAND_MU_C", "COMPAND_MU_E", "COMPAND_A_C", "COMPAND_A_E"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), atol=1e-12, rtol=1e-12)

    # 3. Block coding (exact integer parity vs MATLAB R2025b)
    for k in ["HAMM3_H", "HAMM3_G", "CYCLPOLY_7_4_MIN", "CYCLPOLY_7_4_ALL",
              "CYCLGEN_SYS_H", "CYCLGEN_SYS_G", "CYCLGEN_NOSYS_G", "SYNDTABLE_HAMM3"]:
        check_exact(f"MATLAB:{k}", parse_int(keel_kv[k]), parse_int(mat_kv[k]))

    # 4. Convolutional coding (exact integer parity vs MATLAB R2025b)
    for k in ["TRELLIS_NEXT", "TRELLIS_OUT", "CONVENC_6_7", "VITDEC_6_7"]:
        check_exact(f"MATLAB:{k}", parse_int(keel_kv[k]), parse_int(mat_kv[k]))

    # 5. Source coding (vs MATLAB R2025b)
    check_close("MATLAB:HUFF_AVGLEN", parse_f64(keel_kv["HUFF_AVGLEN"]), parse_f64(mat_kv["HUFF_AVGLEN"]), 1e-14, 1e-14)
    check_exact("MATLAB:ARITH_BITS", parse_int(keel_kv["ARITH_BITS"]), parse_int(mat_kv["ARITH_BITS"]))
    check_exact("MATLAB:QUANTIZ_IDX", parse_int(keel_kv["QUANTIZ_IDX"]), parse_int(mat_kv["QUANTIZ_IDX"]))
    check_close("MATLAB:QUANTIZ_VAL", parse_f64(keel_kv["QUANTIZ_VAL"]), parse_f64(mat_kv["QUANTIZ_VAL"]), 1e-14, 1e-14)
    check_close("MATLAB:QUANTIZ_DIST", parse_f64(keel_kv["QUANTIZ_DIST"]), parse_f64(mat_kv["QUANTIZ_DIST"]), 1e-14, 1e-14)
    check_close("MATLAB:LLOYDS_PART", parse_f64(keel_kv["LLOYDS_PART"]), parse_f64(mat_kv["LLOYDS_PART"]), 1e-5, 1e-5)
    check_close("MATLAB:LLOYDS_CB", parse_f64(keel_kv["LLOYDS_CB"]), parse_f64(mat_kv["LLOYDS_CB"]), 1e-5, 1e-5)
    check_close("MATLAB:DPCMOPT_PRED", parse_f64(keel_kv["DPCMOPT_PRED"]), parse_f64(mat_kv["DPCMOPT_PRED"]), 1e-12, 1e-12)
    check_close("MATLAB:DPCMOPT_CB", parse_f64(keel_kv["DPCMOPT_CB"]), parse_f64(mat_kv["DPCMOPT_CB"]), 1e-5, 1e-5)

    # 6. FSK, OFDM, Analog modulation
    check_close("MATLAB:FSK4_CONT", parse_c64(keel_kv["FSK4_CONT"]), parse_c64(mat_kv["FSK4_CONT"]), 1e-12, 1e-12)
    check_close("MATLAB:FSK4_DISC", parse_c64(keel_kv["FSK4_DISC"]), parse_c64(mat_kv["FSK4_DISC"]), 1e-12, 1e-12)

    # OFDM vs NumPy IFFT + CP
    ofdm_in = np.array([(i % 4) - 1.5 + 1j * ((i // 4) - 1.5) for i in range(16)], dtype=np.complex128).reshape(2, 8)
    ofdm_ref = []
    for s in range(2):
        td = np.fft.ifft(ofdm_in[s])
        ofdm_ref.append(np.concatenate([td[-2:], td]))
    check_close("NUMPY:OFDM_TX", parse_c64(keel_kv["OFDM_TX"]), np.concatenate(ofdm_ref), 1e-12, 1e-12)

    for k in ["PM_MOD", "AM_MOD", "FM_MOD", "SSB_LOWER", "SSB_UPPER"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-12, 1e-12)

    # 7. Channel metrics (vs SciPy + MATLAB R2025b)
    q_x = np.array([-2.0, 0.0, 1.0, 2.5, 5.0])
    check_close("SCIPY:QFUNC", parse_f64(keel_kv["QFUNC"]), 0.5 * sc_special.erfc(q_x / math.sqrt(2.0)), 1e-15, 1e-14)
    check_close("MATLAB:QFUNC", parse_f64(keel_kv["QFUNC"]), parse_f64(mat_kv["QFUNC"]), 1e-15, 1e-14)

    # Note: 8-PSK high-SNR analytical approximation in keel matches MATLAB berawgn closely;
    # check BPSK/QPSK/16-QAM/4-PAM/2-FSK/2-DPSK tightly and 8-PSK to analytical bound tolerance.
    k_ber = parse_f64(keel_kv["BER_AWGN"])
    m_ber = parse_f64(mat_kv["BER_AWGN"])
    for idx, label in enumerate(["BPSK", "QPSK", "8PSK", "16QAM", "4PAM", "2FSK", "2DPSK"]):
        tol = 5e-3 if label == "8PSK" else (1e-9 if label == "4PAM" else 1e-12)
        check_close(f"MATLAB:BER_{label}", np.array([k_ber[idx]]), np.array([m_ber[idx]]), atol=1e-14, rtol=tol)

    check_close("MATLAB:NOISE_BW", parse_f64(keel_kv["NOISE_BW"]), parse_f64(mat_kv["NOISE_BW"]), 1e-11, 1e-11)

    # Extra SciPy independent check for SSB & noisebw Parseval identity
    xsig = 0.4 * np.sin(2.0 * np.pi * 50.0 * np.arange(32) / 1000.0)
    ang = 2.0 * np.pi * 200.0 * np.arange(32) / 1000.0 + 0.2
    ana = sc_signal.hilbert(xsig)
    check_close("SCIPY:SSB_LOWER", parse_f64(keel_kv["SSB_LOWER"]), xsig * np.cos(ang) + np.imag(ana) * np.sin(ang), 1e-12, 1e-12)
    check_close("SCIPY:SSB_UPPER", parse_f64(keel_kv["SSB_UPPER"]), xsig * np.cos(ang) - np.imag(ana) * np.sin(ang), 1e-12, 1e-12)

    print(f"VERIFICATION SUMMARY: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
