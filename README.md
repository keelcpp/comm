# keel::comm

Zero-STL C++20 digital communications kernels built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Digital & analog modulation**: M-PSK (`psk_mod`/`psk_demod`), M-DPSK (`dpsk_mod`/`dpsk_demod`), M-PAM (`pam_mod`/`pam_demod`), rectangular M-QAM (`qam_mod`/`qam_demod`), M-FSK (`fsk_mod`/`fsk_demod`), OFDM (`ofdm_mod`/`ofdm_demod`), multi-ring APSK & generic constellations (`apsk_constellation`, `genqam_mod`/`genqam_demod`), differential MSK (`msk_mod_diff`/`msk_demod_diff`), analog PM/AM/FM/SSB (`pm_mod`/`pm_demod`, `am_mod`, `fm_mod`/`fm_demod`, `ssb_mod`), Gray coding (`to_gray`/`from_gray`), and constellation normalization (`modnorm`).
- **Forward error correction**: GF(2) linear block codes (`default_prim_poly`, `gen2par`, `hammgen`, `cyclpoly`, `cyclgen`, `syndtable`, `block_encode`, `block_decode`), rate-1/n convolutional coding (`poly2trellis`, `convenc`), and hard-decision Viterbi decoding (`vitdec_hard`).
- **Source coding & quantization**: Huffman coding (`huffman_dict`, `huffman_encode`, `huffman_decode`), adaptive-precision integer arithmetic coding (`arith_encode`, `arith_decode`), scalar quantization (`quantiz_index`, `quantiz`), Lloyd-Max quantizer design (`lloyds`), and DPCM / Yule-Walker Levinson-Durbin optimization (`dpcm_encode`, `dpcm_decode`, `dpcm_opt`).
- **Pulse shaping & companding**: raised-cosine and root-raised-cosine FIR design (`rcos_design`), Gaussian pulse-shaping FIR (`gauss_design`), and μ-law / A-law companding (`compand`).
- **Scrambling & error metrics**: multiplicative scrambler/descrambler (`scramble`/`descramble`), bit-error rate (`bit_error_rate`), and symbol-error rate (`sym_error_rate`).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
