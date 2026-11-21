# keel::comm

Zero-STL C++20 digital communications kernels built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Digital modulation & demodulation**: M-PSK (`psk_mod`/`psk_demod`), M-DPSK (`dpsk_mod`/`dpsk_demod`), M-PAM (`pam_mod`/`pam_demod`), rectangular M-QAM (`qam_mod`/`qam_demod`), Gray coding (`to_gray`/`from_gray`), and constellation normalization (`modnorm`).
- **Pulse shaping & companding**: raised-cosine and root-raised-cosine FIR design (`rcos_design`), Gaussian pulse-shaping FIR (`gauss_design`), and μ-law / A-law companding (`compand`).
- **Scrambling & error metrics**: multiplicative scrambler/descrambler (`scramble`/`descramble`), bit-error rate (`bit_error_rate`), and symbol-error rate (`sym_error_rate`).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
