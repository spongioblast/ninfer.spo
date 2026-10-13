#include "models/gemma4/frontend/tokenizer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool expect(const char* label, const std::vector<int>& got, const std::vector<int>& want) {
    if (got == want) { return true; }
    std::cerr << "FAIL " << label << '\n';
    return false;
}

} // namespace

int main() {
    const char* hf_dir = std::getenv("NINFER_GEMMA4_HF");
    std::ifstream input(hf_dir ? std::string(hf_dir) + "/tokenizer.json" : std::string(),
                        std::ios::binary);
    if (!input) {
        std::cout << "SKIP: NINFER_GEMMA4_HF does not hold tokenizer.json\n";
        return 77;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    try {
        const ninfer::models::gemma4::Tokenizer tokenizer(buffer.str());
        int failures = 0;
        if (!expect("france", tokenizer.encode("The capital of France is"),
                    {818, 5279, 529, 7001, 563})) {
            ++failures;
        }
        if (!expect("hello", tokenizer.encode("Hello"), {9259})) { ++failures; }
        if (!expect("spaces", tokenizer.encode("a b"), {236746, 518})) { ++failures; }
        if (!expect("newline", tokenizer.encode("line\nbreak"), {1257, 107, 7284})) { ++failures; }
        if (!expect("emoji", tokenizer.encode("\U0001F600"), {242398})) { ++failures; }
        if (!expect("cjk", tokenizer.encode("\u4e2d"), {237103})) { ++failures; }
        std::cout << (failures ? "FAIL" : "OK") << " gemma4 tokenizer\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
