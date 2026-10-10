#pragma once
// Common byte interpretation for the host oracle and every HIP kernel.
// GGML block formulas follow ggml-org/llama.cpp (MIT); wire offsets, not
// C++ struct packing, are authoritative. Kernel-native groups are little-endian
// fp16 scale + uint8 zero point + a bit stream of G codes.
#include <cstdint>
#include <cstring>
#ifdef __HIPCC__
#define KNJ_HD __host__ __device__
#else
#define KNJ_HD
#endif
namespace knj::decode {
constexpr uint32_t kGroupTag = 0x10000000u;
KNJ_HD inline uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
KNJ_HD inline uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
KNJ_HD inline float from_bits(uint32_t v) { float f; __builtin_memcpy(&f, &v, 4); return f; }
KNJ_HD inline uint32_t bits(float v) { uint32_t b; __builtin_memcpy(&b, &v, 4); return b; }
KNJ_HD inline float half(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    if (e == 31) return from_bits(sign | 0x7f800000u | (m << 13));
    if (e != 0) return from_bits(sign | ((e + 112) << 23) | (m << 13));
    if (m == 0) return from_bits(sign);
    int exponent = -14;
    while ((m & 1024) == 0) { m <<= 1; --exponent; }
    return from_bits(sign | (uint32_t(exponent + 127) << 23) | ((m & 1023) << 13));
}
KNJ_HD inline uint16_t to_half(float f) {
    uint32_t b = bits(f), sign = (b >> 16) & 0x8000u, exp = (b >> 23) & 255, man = b & 0x7fffffu;
    if (exp == 255) return uint16_t(sign | 0x7c00 | (man ? 0x200 : 0));
    int e = int(exp) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        man |= 0x800000u;
        uint32_t shift = uint32_t(14 - e), v = man >> shift, rem = man & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        v += rem > halfway || (rem == halfway && (v & 1));
        return uint16_t(sign | v);
    }
    uint32_t v = (uint32_t(e) << 10) | (man >> 13), rem = man & 8191;
    v += rem > 4096 || (rem == 4096 && (v & 1));
    return uint16_t(sign | v);
}
KNJ_HD inline float fp8(uint8_t c) {
    int e = (c >> 3) & 15, m = c & 7; float v;
    if (e == 15 && m == 7) return from_bits(0x7fc00000u);
    if (e == 0) v = float(m) * (1.0f / 512.0f);
    else v = from_bits(uint32_t(e + 120) << 23) * (1.0f + float(m) * 0.125f);
    return c & 128 ? -v : v;
}
KNJ_HD inline uint8_t to_fp8(float f) {
    uint32_t b = bits(f); uint8_t sign = uint8_t((b >> 24) & 128); float a = from_bits(b & 0x7fffffff);
    if ((b & 0x7fffffff) > 0x7f800000) return uint8_t(sign | 127);
    if (a >= 448.0f) return uint8_t(sign | 126);
    // The 127 finite positive values are monotonic. Binary search followed by
    // ties-to-even implements E4M3FN exactly, including subnormals and -0.
    uint32_t lo = 0, hi = 126;
    while (lo < hi) { uint32_t mid = (lo + hi) / 2; if (fp8(uint8_t(mid)) < a) lo = mid + 1; else hi = mid; }
    if (lo && a - fp8(uint8_t(lo - 1)) < fp8(uint8_t(lo)) - a) --lo;
    else if (lo && a - fp8(uint8_t(lo - 1)) == fp8(uint8_t(lo)) - a && (lo & 1)) --lo;
    return uint8_t(sign | lo);
}
KNJ_HD inline uint32_t group_type(uint32_t b, uint32_t g) { return kGroupTag | (g << 8) | b; }
KNJ_HD inline bool is_group(uint32_t t) { return (t & kGroupTag) != 0; }
KNJ_HD inline uint32_t group_bits(uint32_t t) { return t & 255; }
KNJ_HD inline uint32_t group_size(uint32_t t) { return (t >> 8) & 0xffff; }
KNJ_HD inline uint32_t code(const uint8_t* p, uint32_t i, uint32_t b) {
    uint32_t bit = i * b, shift = bit & 7, v = p[bit >> 3];
    if (shift + b > 8) v |= uint32_t(p[(bit >> 3) + 1]) << 8;
    return (v >> shift) & ((1u << b) - 1);
}
KNJ_HD inline uint32_t block_elements(uint32_t t) {
    if (t & kGroupTag) return (t >> 8) & 0xffff;
    if (t == 2 || t == 3 || t == 6 || t == 7 || t == 8 || t == 9) return 32;
    if (t >= 10 && t <= 15) return 256;
    return 1;
}
KNJ_HD inline uint32_t block_bytes(uint32_t t) {
    if (t & kGroupTag) return 3 + (block_elements(t) * (t & 255) + 7) / 8;
    switch (t) {
        case 0: return 4; case 1: case 30: return 2;
        case 2: return 18; case 3: return 20; case 6: return 22; case 7: return 24;
        case 8: return 34; case 9: return 36; case 10: return 84; case 11: return 110;
        case 12: return 144; case 13: return 176; case 14: return 210; case 15: return 292;
        default: return 0;
    }
}
KNJ_HD inline float weight(uint32_t t, const uint8_t* data, uint64_t index) {
    const uint32_t be = block_elements(t), bb = block_bytes(t);
    const uint8_t* p = data + (index / be) * bb; uint32_t i = uint32_t(index % be);
    if (t & kGroupTag) return half(u16(p)) * float(int(code(p + 3, i, t & 255)) - int(p[2]));
    switch (t) {
        case 0: return from_bits(u32(p));
        case 1: return half(u16(p));
        case 30: return from_bits(uint32_t(u16(p)) << 16);
        case 2: return half(u16(p)) * float(int((p[2 + i % 16] >> (i / 16 * 4)) & 15) - 8);
        case 3: return half(u16(p)) * float((p[4 + i % 16] >> (i / 16 * 4)) & 15) + half(u16(p + 2));
        case 6: case 7: {
            // Q5_0/Q5_1: the 5th bit of element i is qh bit i (low nibbles at bits 0-15, high
            // nibbles at bits 16-31) — see quantize_row_q5_0/q5_1 and dequantize_row_q5_0/q5_1.
            uint32_t start = t == 6 ? 2 : 4;
            int v = int((p[start + 4 + i % 16] >> (i / 16 * 4)) & 15) | int(((u32(p + start) >> i) & 1) << 4);
            return half(u16(p)) * float(v - (t == 6 ? 16 : 0)) + (t == 7 ? half(u16(p + 2)) : 0.0f);
        }
        case 8: return half(u16(p)) * float(int8_t(p[2 + i]));
        case 9: return half(u16(p)) * float(int8_t(p[4 + i]));
        case 10: {
            uint32_t sc = p[i / 16], q = (p[16 + (i / 128) * 32 + i % 32] >> ((i % 128) / 32 * 2)) & 3;
            return half(u16(p + 80)) * float(sc & 15) * float(q) - half(u16(p + 82)) * float(sc >> 4);
        }
        case 11: {
            uint32_t g = i / 16; const uint8_t* s = p + 96;
            int sc = int(((s[g % 8] >> (g / 8 * 4)) & 15) | (((s[8 + g % 4] >> (g / 4 * 2)) & 3) << 4)) - 32;
            int q = int((p[32 + (i / 128) * 32 + i % 32] >> ((i % 128) / 32 * 2)) & 3);
            q -= (p[i % 32] & (1u << (i / 32))) ? 0 : 4;
            return half(u16(p + 108)) * float(sc) * float(q);
        }
        case 12: case 13: {
            uint32_t g = i / 32, sc, mn; const uint8_t* s = p + 4;
            if (g < 4) { sc = s[g] & 63; mn = s[g + 4] & 63; }
            else { sc = (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4); mn = (s[g + 4] >> 4) | ((s[g] >> 6) << 4); }
            uint32_t qoff = t == 12 ? 16 : 48;
            uint32_t q = (p[qoff + (i / 64) * 32 + i % 32] >> ((i % 64) / 32 * 4)) & 15;
            if (t == 13 && (p[16 + i % 32] & (1u << (i / 32)))) q += 16;
            return half(u16(p)) * float(sc) * float(q) - half(u16(p + 2)) * float(mn);
        }
        case 14: {
            uint32_t n = i / 128, j = i % 128;
            int q = int((p[n * 64 + j % 64] >> (j / 64 * 4)) & 15) |
                    int(((p[128 + n * 32 + j % 32] >> (j / 32 * 2)) & 3) << 4);
            return half(u16(p + 208)) * float(int8_t(p[192 + i / 16])) * float(q - 32);
        }
        case 15: return from_bits(u32(p)) * float(int8_t(p[4 + i]));
        default: return from_bits(0x7fc00000u); // unreachable after plan validation
    }
}
}  // namespace knj::decode
#undef KNJ_HD
