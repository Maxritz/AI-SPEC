// Lower-layer read checks: every backend available on this host returns exactly the
// bytes of the file for random positioned reads, including under concurrent use, and
// fails with a typed error for short reads and missing files. Explicit requests for a
// backend that the platform does not provide must fail rather than substitute.
#include "test_support.h"
#include "core/config.h"
#include "core/error.h"
#include "inference/engine.h"
#include "io/reader.h"
#include "synthetic_model.h"

#include <atomic>
#include <fstream>
#include <random>
#include <thread>

using namespace knj;

namespace {

std::vector<uint8_t> pattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> out(n);
    std::mt19937 rng(seed);
    for (auto& b : out) b = uint8_t(rng());
    return out;
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// Random reads, including zero-length, single-byte, page-crossing and end-of-file reads.
void verify_reads(io::Reader& reader, const std::string& path, const std::vector<uint8_t>& truth, uint32_t seed, uint32_t count) {
    std::mt19937 rng(seed);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t max = truth.size();
        const uint64_t offset = rng() % max;
        const uint64_t room = max - offset;
        const uint64_t bytes = std::min<uint64_t>(room, 1 + rng() % (i % 4 == 0 ? 512 * 1024 : 9000));
        std::vector<uint8_t> got(size_t(bytes), 0xAA);
        if (bytes == 0) {
            reader.read(path, offset, got.data(), 0);
            continue;
        }
        reader.read(path, offset, got.data(), size_t(bytes));
        CHECK(std::equal(got.begin(), got.end(), truth.begin() + std::ptrdiff_t(offset)));
    }
    // The last byte alone, and the whole file in one request.
    uint8_t last = 0;
    reader.read(path, truth.size() - 1, &last, 1);
    CHECK(last == truth.back());
    std::vector<uint8_t> whole(truth.size());
    reader.read(path, 0, whole.data(), whole.size());
    CHECK(whole == truth);
}

void verify_errors(io::Reader& reader, const std::string& path, uint64_t size) {
    uint8_t buffer[64];
    // Crossing the end of the file is a short read, not a silent partial result.
    test::throws([&] { reader.read(path, size - 10, buffer, 64); });
    // A missing file is an I/O error.
    test::throws([&] { reader.read(path + ".missing", 0, buffer, 1); });
}

void concurrent_reads(io::Reader& reader, const std::string& path, const std::vector<uint8_t>& truth) {
    std::atomic<uint32_t> failures{0};
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < 6; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(1000 + t);
            for (uint32_t i = 0; i < 120; ++i) {
                const uint64_t offset = rng() % truth.size();
                const uint64_t bytes = std::min<uint64_t>(truth.size() - offset, 1 + rng() % 70000);
                std::vector<uint8_t> got(static_cast<size_t>(bytes));
                try {
                    reader.read(path, offset, got.data(), got.size());
                    if (!std::equal(got.begin(), got.end(), truth.begin() + std::ptrdiff_t(offset))) failures++;
                } catch (const std::exception&) {
                    failures++;
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(failures.load() == 0);
}

// The engine's workers read through the configured backend; "auto" must land on the native
// backend of this platform or on the portable fallback, and report which one it chose.
void engine_reports_backend(const std::string& dir, const std::string& model_path, const std::string& backend) {
    Config c;
    c.backend = "cpu"; c.kv_codec = "f32"; c.store_dir = dir + "/store-" + backend; c.tune_dir = dir + "/tune";
    c.token_block = 16; c.profile_floor = 8; c.io_backend = backend;
    inference::Engine engine(model_path, c);
    const auto io = engine.report().at("io");
    const std::string chosen = io.at("backend").get<std::string>();
    if (backend == "portable") {
        CHECK(chosen == "portable");
        CHECK(!io.at("native").get<bool>());
    } else {
#ifdef __linux__
        CHECK(chosen == "io_uring" || chosen == "portable");
#else
        CHECK(chosen == "iocp" || chosen == "portable");
#endif
        CHECK(io.at("native").get<bool>() == (chosen != "portable"));
    }
    CHECK(io.at("queue_depth").get<uint32_t>() >= 1);
    CHECK(!io.at("detail").get<std::string>().empty());
    std::cout << "  engine io (" << backend << " requested): " << chosen << "\n";
}

}  // namespace

int main() {
    try {
        test::TempDir dir;
        const std::string root = dir.path.string();

        // Names and parsing.
        CHECK(io::parse_preference("auto") == io::Preference::Auto);
        CHECK(io::parse_preference("portable") == io::Preference::Portable);
        CHECK(io::parse_preference("io_uring") == io::Preference::IoUring);
        CHECK(io::parse_preference("iocp") == io::Preference::Iocp);
        test::throws([] { io::parse_preference("uring"); });
        CHECK(std::string(io::preference_name(io::Preference::IoUring)) == "io_uring");
        test::throws([] { io::make_reader(io::Preference::Portable, 0); });
        test::throws([] { io::make_reader(io::Preference::Portable, 4096); });

        // Source bytes: a few MiB plus an odd tail so reads cross pages and blocks.
        const std::string path = root + "/source.bin";
        const std::vector<uint8_t> truth = pattern((6u << 20) + 1234, 77);
        write_bytes(path, truth);

        auto portable = io::make_reader(io::Preference::Portable, 8);
        CHECK(portable->capabilities().backend == "portable");
        CHECK(!portable->capabilities().native);
        verify_reads(*portable, path, truth, 1, 200);
        verify_errors(*portable, path, truth.size());

        // The automatic choice is the platform's native backend when it works.
        auto automatic = io::make_reader(io::Preference::Auto, 32);
        const auto automatic_caps = automatic->capabilities();
        CHECK(automatic_caps.queue_depth >= 1);
        CHECK(!automatic_caps.detail.empty());
        verify_reads(*automatic, path, truth, 2, 200);
        verify_errors(*automatic, path, truth.size());
        concurrent_reads(*automatic, path, truth);

        // Explicit native requests: available means exact; unavailable means a typed error.
#ifdef __linux__
        const io::Preference native = io::Preference::IoUring;
        test::throws([] { io::make_reader(io::Preference::Iocp, 8); });
#else
        const io::Preference native = io::Preference::Iocp;
        test::throws([] { io::make_reader(io::Preference::IoUring, 8); });
#endif
        try {
            auto explicit_native = io::make_reader(native, 16);
            const auto caps = explicit_native->capabilities();
            CHECK(caps.native);
            CHECK(caps.backend == io::preference_name(native));
            CHECK(caps.queue_depth >= 1 && caps.queue_depth <= 16);
            verify_reads(*explicit_native, path, truth, 3, 300);
            verify_errors(*explicit_native, path, truth.size());
            concurrent_reads(*explicit_native, path, truth);
            // Native and portable agree byte for byte on the same ranges.
            std::vector<uint8_t> a(200000), b(200000);
            explicit_native->read(path, 1000, a.data(), a.size());
            portable->read(path, 1000, b.data(), b.size());
            CHECK(a == b);
            std::cout << "  native backend: " << caps.backend << " (" << caps.detail << ")\n";
        } catch (const Error& e) {
            CHECK(e.code() == ErrorCode::Unsupported);
            std::cout << "  native backend unavailable on this host: " << e.what() << "\n";
        }

        // Engine integration: the configured backend is what the workers read through.
        const std::string model_path = root + "/model.gguf";
        synth::write(synth::build(synth::Dims{}, 9), model_path);
        engine_reports_backend(root, model_path, "portable");
        engine_reports_backend(root, model_path, "auto");
        {
            Config bad;
            bad.io_backend = "bogus";
            test::throws([&] { bad.validate(); });
        }
        std::cout << "io checks: " << test::checks << " passed\n";
    } catch (const std::exception& e) {
        std::cerr << "io test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
