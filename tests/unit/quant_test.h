// Test-side GGML quant encoders plus known-answer checks for the production weight decoder.
// The encoders are written to the ggml block layouts (verified against the pinned
// ggml-quants.c dequantizers); the known-answer checks pin the decoder to hand-computed
// values so encoder/decoder drift cannot cancel out. Not used by the engine itself.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <cstring>
#include <string>
#include <vector>

#include "device/weight_decode.h"
#include "test_support.h"
#include "tensor/tensor_io.h"

namespace testquant {

inline void put_u16le(std::string& s, uint16_t v) { s += char(v & 0xff); s += char(uint8_t(v >> 8)); }
inline void put_u32le(std::string& s, uint32_t v) {
    for (int i = 0; i < 4; ++i) s += char(uint8_t(v >> (8 * i)));
}
inline void put_f32le(std::string& s, float f) { uint32_t b; std::memcpy(&b, &f, 4); put_u32le(s, b); }
inline void put_f16le(std::string& s, float f) { put_u16le(s, knj::decode::to_half(f)); }

inline int clamp_round(float x, int lo, int hi) {
    long v = std::lround(double(x));
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return int(v);
}

// Encode one block (block_elems(type) floats) to its GGML payload bytes.
inline std::string encode_block(uint32_t type, const float* x, uint32_t n);

inline std::string encode_block_f32(const float* x, uint32_t n) {
    std::string s;
    for (uint32_t i = 0; i < n; ++i) put_f32le(s, x[i]);
    return s;
}
inline std::string encode_block_f16(const float* x, uint32_t n) {
    std::string s;
    for (uint32_t i = 0; i < n; ++i) put_f16le(s, x[i]);
    return s;
}
inline std::string encode_block_bf16(const float* x, uint32_t n) {
    std::string s;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t b;
        std::memcpy(&b, &x[i], 4);
        put_u16le(s, uint16_t(b >> 16));
    }
    return s;
}

// 32-element blocks. q nibbles pack as qs[j] = q[j] | (q[j+16] << 4), matching dequantize_row_q4_0.
inline std::string encode_block_q4_0(const float* x, uint32_t n) {
    float mx = 0;
    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i]));
    const float d = mx > 0 ? mx / 7.0f : 1.0f;
    std::string s;
    put_f16le(s, d);
    for (uint32_t j = 0; j < 16; ++j) {
        int q0 = clamp_round(x[j] / d, -8, 7) + 8, q1 = clamp_round(x[j + 16] / d, -8, 7) + 8;
        s += char(uint8_t(q0 | (q1 << 4)));
    }
    return s;
}
inline std::string encode_block_q4_1(const float* x, uint32_t n) {
    float mn = x[0], mx = x[0];
    for (uint32_t i = 0; i < n; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
    const float d = mx > mn ? (mx - mn) / 15.0f : 1.0f;
    std::string s;
    put_f16le(s, d);
    put_f16le(s, mn);
    for (uint32_t j = 0; j < 16; ++j) {
        int q0 = clamp_round((x[j] - mn) / d, 0, 15), q1 = clamp_round((x[j + 16] - mn) / d, 0, 15);
        s += char(uint8_t(q0 | (q1 << 4)));
    }
    return s;
}
// Q5_0/Q5_1: the 5th bit of element i is qh bit i (low nibbles at bits 0-15, high nibbles at 16-31).
inline std::string encode_block_q5(const float* x, uint32_t n, bool with_min) {
    float mn = 0, mx = 0;
    if (with_min) {
        mn = x[0]; mx = x[0];
        for (uint32_t i = 0; i < n; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
    } else {
        for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i]));
    }
    const float d = with_min ? (mx > mn ? (mx - mn) / 31.0f : 1.0f) : (mx > 0 ? mx / 15.0f : 1.0f);
    uint32_t qh = 0;
    std::vector<int> q(n);
    for (uint32_t i = 0; i < n; ++i) {
        q[i] = with_min ? clamp_round((x[i] - mn) / d, 0, 31) : clamp_round(x[i] / d, -16, 15) + 16;
        if (q[i] & 16) qh |= 1u << i;
    }
    std::string s;
    put_f16le(s, d);
    if (with_min) put_f16le(s, mn);
    put_u32le(s, qh);
    for (uint32_t j = 0; j < 16; ++j) s += char(uint8_t((q[j] & 15) | ((q[j + 16] & 15) << 4)));
    return s;
}
inline std::string encode_block_q8_0(const float* x, uint32_t n) {
    float mx = 0;
    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i]));
    const float d = mx > 0 ? mx / 127.0f : 1.0f;
    std::string s;
    put_f16le(s, d);
    for (uint32_t i = 0; i < n; ++i) s += char(int8_t(clamp_round(x[i] / d, -127, 127)));
    return s;
}
inline std::string encode_block_q8_1(const float* x, uint32_t n) {
    float mx = 0;
    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i]));
    const float d = mx > 0 ? mx / 127.0f : 1.0f;
    std::string s;
    put_f16le(s, d);
    put_f16le(s, 0.0f);  // s (sum of quants) is not used by dequantization
    for (uint32_t i = 0; i < n; ++i) s += char(int8_t(clamp_round(x[i] / d, -127, 127)));
    return s;
}

// 256-element super-blocks (QK_K). Sub-block scales are stored quantized; the super-block
// scales d/dmin are 1.0 so the stored 6-bit codes are the effective scales.
inline std::string encode_block_q2_k(const float* x, uint32_t /*n*/) {
    std::string s(16, '\0');  // scales[16]: low nibble = scale, high nibble = min
    std::string qs(64, '\0');
    for (uint32_t sub = 0; sub < 16; ++sub) {
        const float* b = x + sub * 16;
        float mn = b[0], mx = b[0];
        for (int i = 0; i < 16; ++i) { mn = std::min(mn, b[i]); mx = std::max(mx, b[i]); }
        int sc = clamp_round((mx - mn) / 3.0f, 1, 15), mnc = clamp_round(-mn, 0, 15);
        s[sub] = char(uint8_t(sc | (mnc << 4)));
        for (int i = 0; i < 16; ++i) {
            int q = clamp_round((b[i] - mn) / float(sc), 0, 3);
            // element (sub*16 + i): within its 128-chunk, byte = (sub%2)*16 + i, shift = (sub/2)*2
            uint32_t byte = (sub % 2) * 16 + uint32_t(i) + (sub / 8) * 32;
            uint32_t shift = (sub / 2 % 4) * 2;
            qs[byte] = char(uint8_t(qs[byte]) | (q << shift));
        }
    }
    put_f16le(s, 1.0f);
    put_f16le(s, 1.0f);
    s = s.substr(0, 16) + qs + s.substr(16);
    return s;
}
inline std::string encode_block_q3_k(const float* x, uint32_t /*n*/) {
    std::string hm(32, '\0'), qs(64, '\0');
    std::vector<int> s6(16, 33);  // 6-bit scales, biased by 32 (effective scale = s6 - 32)
    for (uint32_t sub = 0; sub < 16; ++sub) {
        const float* b = x + sub * 16;
        float mx = 0;
        for (int i = 0; i < 16; ++i) mx = std::max(mx, std::fabs(b[i]));
        int eff = clamp_round(mx / 3.0f, 1, 31);
        s6[sub] = 32 + eff;
        for (int i = 0; i < 16; ++i) {
            int q = clamp_round(b[i] / float(eff), -4, 3);
            uint32_t e = sub * 16 + uint32_t(i);
            uint32_t byte = (e / 128) * 32 + (sub % 2) * 16 + uint32_t(i);
            uint32_t shift = (sub / 2 % 4) * 2;
            qs[byte] = char(uint8_t(qs[byte]) | ((q & 3) << shift));
            if (q >= 0) hm[e % 32] = char(uint8_t(hm[e % 32]) | (1u << (e / 32)));
        }
    }
    // scales[12]: invert the aux transformation of dequantize_row_q3_K
    std::string scales(12, '\0');
    for (int j = 0; j < 4; ++j) {
        scales[j] = char(uint8_t((s6[j] & 15) | ((s6[j + 8] & 15) << 4)));
        scales[j + 4] = char(uint8_t((s6[j + 4] & 15) | ((s6[j + 12] & 15) << 4)));
        scales[j + 8] = char(uint8_t(((s6[j] >> 4) & 3) | (((s6[j + 4] >> 4) & 3) << 2) |
                                      (((s6[j + 8] >> 4) & 3) << 4) | (((s6[j + 12] >> 4) & 3) << 6)));
    }
    std::string s = hm + qs + scales;
    put_f16le(s, 1.0f);
    return s;
}
// get_scale_min_k4 packing: scales[0..3] = sc | (sc' >> 4 << 6), scales[4..7] = mn | (mn' >> 4 << 6),
// scales[8..11] = (sc' & 15) | ((mn' & 15) << 4) for the second group of four sub-blocks.
inline void pack_scales_k4(std::string& s, const std::vector<int>& sc, const std::vector<int>& mn) {
    for (int j = 0; j < 4; ++j) {
        s[j] = char(uint8_t(sc[j] | ((sc[j + 4] >> 4) << 6)));
        s[j + 4] = char(uint8_t(mn[j] | ((mn[j + 4] >> 4) << 6)));
        s[j + 8] = char(uint8_t((sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4)));
    }
}
inline std::string encode_block_q4_k(const float* x, uint32_t /*n*/) {
    std::vector<int> sc(8, 1), mn(8, 0);
    std::string qs(128, '\0');
    for (uint32_t sub = 0; sub < 8; ++sub) {
        const float* b = x + sub * 32;
        float lo = b[0], hi = b[0];
        for (int i = 0; i < 32; ++i) { lo = std::min(lo, b[i]); hi = std::max(hi, b[i]); }
        sc[sub] = clamp_round((hi - lo) / 15.0f, 1, 63);
        mn[sub] = clamp_round(-lo, 0, 63);
        for (int i = 0; i < 32; ++i) {
            int q = clamp_round((b[i] - lo) / float(sc[sub]), 0, 15);
            uint32_t byte = (sub / 2) * 32 + uint32_t(i);
            uint32_t shift = (sub % 2) * 4;
            qs[byte] = char(uint8_t(qs[byte]) | (q << shift));
        }
    }
    std::string s;
    put_f16le(s, 1.0f);
    put_f16le(s, 1.0f);
    std::string scales(12, '\0');
    pack_scales_k4(scales, sc, mn);
    s += scales + qs;
    return s;
}
inline std::string encode_block_q5_k(const float* x, uint32_t /*n*/) {
    std::vector<int> sc(8, 1), mn(8, 0);
    std::string qh(32, '\0'), qs(128, '\0');
    for (uint32_t sub = 0; sub < 8; ++sub) {
        const float* b = x + sub * 32;
        float lo = b[0], hi = b[0];
        for (int i = 0; i < 32; ++i) { lo = std::min(lo, b[i]); hi = std::max(hi, b[i]); }
        sc[sub] = clamp_round((hi - lo) / 31.0f, 1, 63);
        mn[sub] = clamp_round(-lo, 0, 63);
        for (int i = 0; i < 32; ++i) {
            int q = clamp_round((b[i] - lo) / float(sc[sub]), 0, 31);
            uint32_t e = sub * 32 + uint32_t(i);
            uint32_t byte = (sub / 2) * 32 + uint32_t(i);
            uint32_t shift = (sub % 2) * 4;
            qs[byte] = char(uint8_t(qs[byte]) | ((q & 15) << shift));
            if (q & 16) qh[e % 32] = char(uint8_t(qh[e % 32]) | (1u << (e / 32)));
        }
    }
    std::string s;
    put_f16le(s, 1.0f);
    put_f16le(s, 1.0f);
    std::string scales(12, '\0');
    pack_scales_k4(scales, sc, mn);
    s += scales + qh + qs;
    return s;
}
inline std::string encode_block_q6_k(const float* x, uint32_t /*n*/) {
    std::string ql(128, '\0'), qh(64, '\0'), sc(16, '\0');
    for (uint32_t sub = 0; sub < 16; ++sub) {
        const float* b = x + sub * 16;
        float mx = 0;
        for (int i = 0; i < 16; ++i) mx = std::max(mx, std::fabs(b[i]));
        int eff = clamp_round(mx / 31.0f, 1, 127);
        sc[sub] = char(int8_t(eff));
        for (int i = 0; i < 16; ++i) {
            int q = clamp_round(b[i] / float(eff), -32, 31) + 32;  // 6-bit code
            uint32_t e = sub * 16 + uint32_t(i);
            uint32_t j = e % 128;
            uint32_t byte = (e / 128) * 64 + j % 64;
            uint32_t shift = (j / 64) * 4;
            ql[byte] = char(uint8_t(ql[byte]) | ((q & 15) << shift));
            uint32_t hbyte = (e / 128) * 32 + j % 32;  // index within qh (payload offset 128 + hbyte)
            uint32_t hshift = (j / 32) * 2;
            qh[hbyte] = char(uint8_t(qh[hbyte]) | (((q >> 4) & 3) << hshift));
        }
    }
    std::string s = ql + qh + sc;
    put_f16le(s, 1.0f);
    return s;
}
inline std::string encode_block_q8_k(const float* x, uint32_t n) {
    float mx = 0;
    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(x[i]));
    const float d = mx > 0 ? mx / 127.0f : 1.0f;
    std::string s;
    put_f32le(s, d);
    for (uint32_t i = 0; i < n; ++i) s += char(int8_t(clamp_round(x[i] / d, -127, 127)));
    s.append(32, '\0');  // bsums: unused by dequantization
    return s;
}

inline std::string encode_block(uint32_t type, const float* x, uint32_t n) {
    switch (type) {
        case 0: return encode_block_f32(x, n);
        case 1: return encode_block_f16(x, n);
        case 30: return encode_block_bf16(x, n);
        case 2: return encode_block_q4_0(x, n);
        case 3: return encode_block_q4_1(x, n);
        case 6: return encode_block_q5(x, n, false);
        case 7: return encode_block_q5(x, n, true);
        case 8: return encode_block_q8_0(x, n);
        case 9: return encode_block_q8_1(x, n);
        case 10: return encode_block_q2_k(x, n);
        case 11: return encode_block_q3_k(x, n);
        case 12: return encode_block_q4_k(x, n);
        case 13: return encode_block_q5_k(x, n);
        case 14: return encode_block_q6_k(x, n);
        case 15: return encode_block_q8_k(x, n);
        default: throw std::runtime_error("no test encoder for ggml type " + std::to_string(type));
    }
}

// Encode a whole tensor payload. `x` must hold a multiple of the block size.
inline std::string encode(uint32_t type, const std::vector<float>& x) {
    const uint32_t be = knj::decode::block_elements(type);
    if (x.size() % be) throw std::runtime_error("tensor element count is not a multiple of the block size");
    std::string s;
    for (size_t i = 0; i < x.size(); i += be) s += encode_block(type, x.data() + i, be);
    return s;
}

// Known-answer checks: literal block bytes with hand-computed dequantized values, pinning the
// production decoder (knj::decode::weight) to the ggml-quants.c semantics.
inline void known_answer_checks() {
    using knj::decode::weight;
    auto w = [](uint32_t t, const std::string& b, uint64_t i) { return weight(t, reinterpret_cast<const uint8_t*>(b.data()), i); };
    // F32 / F16 / BF16 (each value is its own one-element block at its own offset)
    {
        std::string b;
        put_f32le(b, 1.5f);       // offset 0
        put_f16le(b, 1.0f);       // offset 4
        put_f16le(b, -2.0f);      // offset 6
        put_u16le(b, 0x3f80);     // offset 8:  BF16 1.0
        put_u16le(b, 0xc000);     // offset 10: BF16 -2.0
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b.data());
        CHECK(weight(0, p, 0) == 1.5f);
        CHECK(weight(1, p + 4, 0) == 1.0f);
        CHECK(weight(1, p + 6, 0) == -2.0f);
        CHECK(weight(30, p + 8, 0) == 1.0f);
        CHECK(weight(30, p + 10, 0) == -2.0f);
    }
    // Q4_0: d=2, qs[0]=0x21 -> elem0 = (1-8)*2 = -14, elem16 = (2-8)*2 = -12
    {
        std::string b;
        put_f16le(b, 2.0f);
        for (int j = 0; j < 16; ++j) b += char(j == 0 ? 0x21 : 0x00);
        CHECK(w(2, b, 0) == -14.0f);
        CHECK(w(2, b, 16) == -12.0f);
        CHECK(w(2, b, 1) == -16.0f);
    }
    // Q4_1: d=0.5, m=1, qs[0]=0x21 -> elem0 = 1*0.5+1 = 1.5, elem16 = 2*0.5+1 = 2
    {
        std::string b;
        put_f16le(b, 0.5f);
        put_f16le(b, 1.0f);
        for (int j = 0; j < 16; ++j) b += char(j == 0 ? 0x21 : 0x00);
        CHECK(w(3, b, 0) == 1.5f);
        CHECK(w(3, b, 16) == 2.0f);
    }
    // Q5_0: d=1, qh bit 16 set, qs[j]=0x10 -> elem j (j<16) = (0|0)-16 = -16, elem16 = (1|16)-16 = 1.
    // The 5th bit of element 16 is qh bit 16 (quantize_row_q5_0: qh bit j+QK5_0/2 for element j+16).
    {
        std::string b;
        put_f16le(b, 1.0f);
        put_u32le(b, 0x00010000u);
        for (int j = 0; j < 16; ++j) b += char(0x10);
        CHECK(w(6, b, 0) == -16.0f);
        CHECK(w(6, b, 16) == 1.0f);
    }
    // Q5_1: d=1, m=0.5, qh bit 16 set, qs[j]=0x10 -> elem0 = 0*1+0.5 = 0.5, elem16 = 17*1+0.5 = 17.5
    {
        std::string b;
        put_f16le(b, 1.0f);
        put_f16le(b, 0.5f);
        put_u32le(b, 0x00010000u);
        for (int j = 0; j < 16; ++j) b += char(0x10);
        CHECK(w(7, b, 0) == 0.5f);
        CHECK(w(7, b, 16) == 17.5f);
    }
    // Q8_0: d=0.25, qs = {1, -2} -> 0.25, -0.5. Q8_1: d=2, qs = {3, -1} -> 6, -2 (s ignored).
    {
        std::string b;
        put_f16le(b, 0.25f);
        for (int j = 0; j < 32; ++j) b += char(int8_t(j == 0 ? 1 : (j == 1 ? -2 : 0)));
        CHECK(w(8, b, 0) == 0.25f);
        CHECK(w(8, b, 1) == -0.5f);
        std::string b2;
        put_f16le(b2, 2.0f);
        put_f16le(b2, 123.0f);  // s: ignored by dequantization
        for (int j = 0; j < 32; ++j) b2 += char(int8_t(j == 0 ? 3 : (j == 1 ? -1 : 0)));
        CHECK(w(9, b2, 0) == 6.0f);
        CHECK(w(9, b2, 1) == -2.0f);
    }
    // Q2_K: scales[0]=0x21 (sc=1, mn=2), scales[1]=0, d=dmin=1, qs[0]=0x55 -> q=1.
    // elem0 = 1*1*1 - 1*2 = -1; elem16 (scales[1]) = 0; elem32 (q shift 2, scales[2]=0) = 0.
    {
        std::string b(16, '\0');
        b[0] = char(0x21);
        std::string qs(64, '\0');
        for (int j = 0; j < 64; ++j) qs[j] = char(0x55);
        put_f16le(b, 1.0f);
        put_f16le(b, 1.0f);
        b = b.substr(0, 16) + qs + b.substr(16);
        CHECK(w(10, b, 0) == -1.0f);
        CHECK(w(10, b, 16) == 0.0f);
        CHECK(w(10, b, 32) == 0.0f);
    }
    // Q3_K: all 6-bit scales = 33 (effective 1), d=1, hm[0] bit 0 set, qs[0]=0x01 -> elem0 = 1*(1-0) = 1.
    // scales packing: scales[j] = 1 | (1<<4) = 17 (j<8), scales[8+j] = 2|8|32|128 = 170.
    {
        std::string hm(32, '\0');
        hm[0] = char(1);
        std::string qs(64, '\0');
        qs[0] = char(0x01);
        std::string scales(12, '\0');
        for (int j = 0; j < 8; ++j) scales[j] = char(17);
        for (int j = 0; j < 4; ++j) scales[j + 8] = char(170);
        std::string b = hm + qs + scales;
        put_f16le(b, 1.0f);
        CHECK(w(11, b, 0) == 1.0f);
        // elem16: scales[1] = 33 -> 1; qs[16] = 0; hm[16] bit 0 clear -> q = 0 - 4 = -4
        CHECK(w(11, b, 16) == -4.0f);
    }
    // Q4_K: d=dmin=1, sc=15, mn=2 everywhere, qs[0]=0x21 -> elem0 = 15*1-2 = 13, elem32 = 15*2-2 = 28.
    {
        std::string b;
        put_f16le(b, 1.0f);
        put_f16le(b, 1.0f);
        std::string scales(12, '\0');
        std::vector<int> sc(8, 15), mn(8, 2);
        pack_scales_k4(scales, sc, mn);
        b += scales;
        std::string qs(128, '\0');
        qs[0] = char(0x21);
        b += qs;
        CHECK(w(12, b, 0) == 13.0f);
        CHECK(w(12, b, 32) == 28.0f);
    }
    // Q5_K: like Q4_K with qh bit 0 set -> elem0 = 15*(1+16)-2 = 253; elem32 = 15*(2+0)-2 = 28.
    {
        std::string b;
        put_f16le(b, 1.0f);
        put_f16le(b, 1.0f);
        std::string scales(12, '\0');
        std::vector<int> sc(8, 15), mn(8, 2);
        pack_scales_k4(scales, sc, mn);
        b += scales;
        std::string qh(32, '\0');
        qh[0] = char(1);  // 5th bit of element 0 (chunk 0, u1 = 1)
        b += qh;
        std::string qs(128, '\0');
        qs[0] = char(0x21);
        b += qs;
        CHECK(w(13, b, 0) == 253.0f);
        CHECK(w(13, b, 32) == 28.0f);
    }
    // Q6_K: sc=2 everywhere, ql[0]=0x05, qh[0] bits 0-1 = 1 -> q = (5 | 16) - 32 = -11 -> elem0 = 2*(-11) = -22.
    // elem64 (q3): ql[0] high nibble = 0, qh[0] bits 4-5 = 0 -> q = -32 -> elem64 = 2*(-32) = -64.
    {
        std::string ql(128, '\0'), qh(64, '\0'), sc(16, '\0');
        ql[0] = char(0x05);
        qh[0] = char(0x01);
        for (int j = 0; j < 16; ++j) sc[j] = char(int8_t(2));
        std::string b = ql + qh + sc;
        put_f16le(b, 1.0f);
        CHECK(w(14, b, 0) == -22.0f);
        CHECK(w(14, b, 64) == -64.0f);
    }
    // Q8_K: d=0.5 (f32), qs[0]=3 -> 1.5, qs[1]=-2 -> -1.
    {
        std::string b;
        put_f32le(b, 0.5f);
        for (int j = 0; j < 256; ++j) b += char(int8_t(j == 0 ? 3 : (j == 1 ? -2 : 0)));
        b.append(32, '\0');
        CHECK(w(15, b, 0) == 1.5f);
        CHECK(w(15, b, 1) == -1.0f);
    }
    // Group quant (engine-internal packed format): half scale, uint8 zero, packed b-bit codes.
    {
        const uint32_t t = knj::decode::group_type(4, 32);
        std::string b;
        put_f16le(b, 2.0f);
        b += char(1);  // zero point
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = 0;
            for (int k = 0; k < 2; ++k) byte |= uint8_t((2 * i + k) & 15) << (4 * k);
            b += char(byte);
        }
        CHECK(w(t, b, 0) == 2.0f * (0 - 1));
        CHECK(w(t, b, 1) == 2.0f * (1 - 1));
        CHECK(w(t, b, 31) == 2.0f * (15 - 1));
    }
    // Round trip: encode random data with the test encoders, decode with the production
    // decoder, and compare against the quantization model (x ~= dequantized within one step).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    for (uint32_t type : {0u, 1u, 30u, 2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}) {
        const uint32_t be = knj::decode::block_elements(type);
        std::vector<float> x(be * 4);
        for (auto& v : x) v = 3.0f * unit(rng);
        const std::string payload = encode(type, x);
        std::vector<float> out(x.size());
        knj::tensor::dequantize(type, reinterpret_cast<const uint8_t*>(payload.data()), payload.size(), x.size(), out.data());
        double worst = 0, magnitude = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, double(std::fabs(out[i] - x[i])));
            magnitude = std::max(magnitude, double(std::fabs(x[i])));
        }
        // F32 must round-trip exactly; every quantized format is checked against a generous
        // ballpark bound (the precise per-format semantics are pinned by the known-answer
        // blocks above; this catches payload size/layout mistakes end to end).
        const double bound = type == 0 ? 1e-6 : 0.6 * (1.0 + magnitude);
        if (worst > bound) std::cerr << "quant round-trip drift type=" << type << " worst=" << worst << '\n';
        CHECK(worst <= bound);
    }
}

}  // namespace testquant
