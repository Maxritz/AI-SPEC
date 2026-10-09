#pragma once
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include "platform/platform.h"
namespace test {
inline uint64_t checks = 0;
inline void check(bool ok, const char* expression, const char* file, int line) {
    ++checks; if (!ok) throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " check failed: " + expression);
}
template<class F> void throws(F f) { ++checks; try { f(); } catch (const std::exception&) { return; } throw std::runtime_error("expected an exception"); }
inline bool close(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol * (1 + std::abs(a)); }
class TempDir {
public:
    std::filesystem::path path;
    TempDir() { path = std::filesystem::temp_directory_path() / ("knj-test-" + std::to_string(knj::platform::now_ns())); std::filesystem::create_directories(path); }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}
#define CHECK(x) test::check(bool(x), #x, __FILE__, __LINE__)
