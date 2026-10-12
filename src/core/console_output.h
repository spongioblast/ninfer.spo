#pragma once

namespace ninfer {

// When stdout is a console whose output code page is not UTF-8, sets it to UTF-8 and restores the
// previous page at destruction. The page belongs to the console, so a child that leaves it changed
// changes the parent too. A redirected stdout is left alone.
class ConsoleOutputCodePage {
public:
    ConsoleOutputCodePage() noexcept;
    ~ConsoleOutputCodePage();

    ConsoleOutputCodePage(const ConsoleOutputCodePage&)            = delete;
    ConsoleOutputCodePage& operator=(const ConsoleOutputCodePage&) = delete;

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    bool active_         = false;
    unsigned previous_   = 0;
};

} // namespace ninfer
