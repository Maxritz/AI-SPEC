// kanjoos-inspect: prints the model index for a GGUF file without loading any
// tensor payload. Usage: kanjoos-inspect <model.gguf> [--experts]
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "gguf/model_index.h"

using namespace knj::gguf;

static std::string human(uint64_t b) {
    char buf[64];
    if (b >= (1ull << 30)) std::snprintf(buf, sizeof buf, "%.2f GiB", double(b) / double(1ull << 30));
    else if (b >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.2f MiB", double(b) / double(1ull << 20));
    else std::snprintf(buf, sizeof buf, "%llu B", (unsigned long long)b);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: kanjoos-inspect <model.gguf> [--experts]\n";
        return 2;
    }
    bool show_experts = argc > 2 && std::strcmp(argv[2], "--experts") == 0;
    try {
        ModelIndex idx = load_model_index(argv[1]);
        const ModelGeometry& g = idx.geometry;
        std::cout << "file:            " << idx.path << "\n";
        std::cout << "file size:       " << human(idx.file_size) << "\n";
        std::cout << "fingerprint:     " << idx.fingerprint << "\n";
        std::cout << "architecture:    " << g.architecture << "\n";
        std::cout << "layers:          " << g.n_layer << "\n";
        std::cout << "embedding:       " << g.n_embd << "\n";
        std::cout << "heads (q/kv):    " << g.n_head << " / " << g.n_head_kv << "  head_dim " << g.head_dim << "\n";
        std::cout << "kv bytes/token:  " << g.kv_bytes_per_token(2) << " (fp16)\n";
        std::cout << "moe:             " << (g.is_moe ? "yes" : "no");
        if (g.is_moe) std::cout << "  experts " << g.n_expert << ", used/token " << g.n_expert_used << ", MoE layers " << idx.experts.size();
        std::cout << "\n";
        std::cout << "alignment:       " << idx.alignment << "\n";
        std::cout << "data offset:     " << idx.data_offset << "\n";
        std::cout << "tensors:         " << idx.tensors.size() << "\n";
        std::cout << "trunk bytes:     " << human(idx.trunk_bytes) << " (resident)\n";
        std::cout << "expert bytes:    " << human(idx.expert_bytes) << " (cold, not read at load)\n";
        std::cout << "bytes read:      " << human(idx.bytes_read_at_load) << " (header only)\n";
        if (show_experts && g.is_moe) {
            for (const ExpertLayer& l : idx.experts) {
                for (int k = 0; k < 3; ++k) {
                    const ExpertTensor& t = l.tensors[k];
                    std::cout << "  layer " << l.layer << " " << expert_kind_name(ExpertKind(k))
                              << ": offset " << t.span.abs_offset << ", " << human(t.bytes_per_expert)
                              << "/expert, " << human(t.span.nbytes) << " total\n";
                }
            }
        }
        return 0;
    } catch (const ParseError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
