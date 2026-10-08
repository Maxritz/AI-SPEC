// kanjoos-store: builds / opens the expert store for a GGUF and reports it.
// Usage: kanjoos-store <model.gguf> [--dir DIR] [--rebuild] [--verify] [--extents]
//                      [--dump LAYER EXPERT OUTFILE]
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "gguf/model_index.h"
#include "store/expert_store.h"

using namespace knj;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: kanjoos-store <model.gguf> [--dir DIR] [--rebuild] [--verify] [--extents]\n";
        return 2;
    }
    std::string model = argv[1];
    std::string dir = model + ".kanjoos";
    bool rebuild = false, verify = false, extents = false;
    bool dump = false;
    uint32_t dump_layer = 0, dump_expert = 0;
    std::string dump_path;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--dump") && i + 3 < argc) {
            dump = true;
            dump_layer = uint32_t(std::stoul(argv[++i]));
            dump_expert = uint32_t(std::stoul(argv[++i]));
            dump_path = argv[++i];
        }
        else if (!std::strcmp(argv[i], "--rebuild")) rebuild = true;
        else if (!std::strcmp(argv[i], "--verify")) verify = true;
        else if (!std::strcmp(argv[i], "--extents")) extents = true;
        else {
            std::cerr << "unknown option: " << argv[i] << "\n";
            return 2;
        }
    }
    try {
        gguf::ModelIndex idx = gguf::load_model_index(model);
        store::ExpertStore st = store::ExpertStore::open(
            dir, idx, store::StoreConfig{}, rebuild ? store::OpenMode::RebuildIfStale : store::OpenMode::Strict);
        std::cout << "store:        " << st.directory() << "\n";
        std::cout << "identity:     " << st.identity() << "\n";
        std::cout << "generation:   " << st.generation() << "\n";
        std::vector<store::Extent> ex = st.extents();
        uint64_t bytes = 0;
        for (const auto& e : ex) bytes += e.nbytes;
        std::cout << "extents:      " << ex.size() << " (" << bytes << " payload bytes)\n";
        if (extents) {
            for (const auto& e : ex) {
                std::cout << "  " << e.name.substr(0, 16) << "  layer " << e.layer << " experts [" << e.first << ", "
                          << e.first + e.count << ")  " << e.nbytes << " B\n";
            }
        }
        if (verify) {
            size_t repaired = st.verify_all();
            std::cout << "verify:       " << repaired << " object(s) repaired\n";
        }
        if (dump) {
            store::ExpertPayload p = st.read_expert(gguf::ExpertId{dump_layer, dump_expert});
            std::FILE* f = std::fopen(dump_path.c_str(), "wb");
            if (!f || std::fwrite(p.bytes.data(), 1, p.bytes.size(), f) != p.bytes.size()) {
                std::cerr << "cannot write " << dump_path << "\n";
                return 1;
            }
            std::fclose(f);
            std::cout << "dumped:       layer " << dump_layer << " expert " << dump_expert << " ("
                      << p.bytes.size() << " B)\n";
        }
        store::Counters c = st.counters();
        std::cout << "counters:     published=" << c.objects_published << " checksum_failures=" << c.checksum_failures
                  << " quarantined=" << c.quarantined_objects << " reread=" << c.reread_from_source
                  << " orphans=" << c.orphans_discarded << " index_rebuilt=" << c.index_rebuilt
                  << " stale_rebuilds=" << c.stale_rebuilds << "\n";
        return 0;
    } catch (const store::StoreError& e) {
        std::cerr << "store error: " << e.what() << "\n";
        return 4;
    } catch (const gguf::ParseError& e) {
        std::cerr << "gguf error: " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
