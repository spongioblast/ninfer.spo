#include "core/console_output.h"

#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    const UINT before = GetConsoleOutputCP();
    const HANDLE out  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode        = 0;
    const bool console =
        out != nullptr && out != INVALID_HANDLE_VALUE && GetConsoleMode(out, &mode) != 0;
    {
        const ninfer::ConsoleOutputCodePage guard;
        const UINT during = GetConsoleOutputCP();
        if (console && before != CP_UTF8) {
            if (!guard.active() || during != CP_UTF8) {
                std::cerr << "stdout console was not switched to UTF-8\n";
                return 1;
            }
        } else if (guard.active() || during != before) {
            std::cerr << "console code page changed when stdout was not a non-UTF-8 console\n";
            return 1;
        }
    }
    if (GetConsoleOutputCP() != before) {
        std::cerr << "console output code page was not restored\n";
        return 1;
    }
#else
    const ninfer::ConsoleOutputCodePage guard;
    if (guard.active()) {
        std::cerr << "console code page guard activated off Windows\n";
        return 1;
    }
#endif
    std::cout << "console code page ok\n";
    return 0;
}
