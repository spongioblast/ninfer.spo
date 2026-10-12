#include "core/console_output.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <atomic>

namespace ninfer {

#ifdef _WIN32
namespace {

std::atomic<UINT> restore_code_page{0};

// The default Ctrl+C handler ends the process without destructors. Put the parent console's
// code page back first, then return FALSE so the next handler (or the default exit) still runs.
BOOL WINAPI restore_on_console_event(DWORD) {
    if (const UINT page = restore_code_page.load(); page != 0) { SetConsoleOutputCP(page); }
    return FALSE;
}

} // namespace
#endif

ConsoleOutputCodePage::ConsoleOutputCodePage() noexcept {
#ifdef _WIN32
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out == nullptr || out == INVALID_HANDLE_VALUE) { return; }
    DWORD mode = 0;
    if (!GetConsoleMode(out, &mode)) { return; }
    const UINT current = GetConsoleOutputCP();
    if (current == CP_UTF8) { return; }
    if (SetConsoleOutputCP(CP_UTF8)) {
        previous_ = current;
        active_   = true;
        restore_code_page.store(current);
        SetConsoleCtrlHandler(restore_on_console_event, TRUE);
    }
#endif
}

ConsoleOutputCodePage::~ConsoleOutputCodePage() {
#ifdef _WIN32
    if (active_) {
        SetConsoleCtrlHandler(restore_on_console_event, FALSE);
        restore_code_page.store(0);
        SetConsoleOutputCP(previous_);
    }
#endif
}

} // namespace ninfer
