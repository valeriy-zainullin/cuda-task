#pragma once

#include <functional>

#define CONCAT2(x, y) x ## y
#define CONCAT(x, y) CONCAT2(x, y)

#define ON_EXIT(...) struct CONCAT(OnExit_, __LINE__) { CONCAT(OnExit_, __LINE__) (std::function<void()>&& callable): callable_(std::move(callable)) {}; ~CONCAT(OnExit_, __LINE__) () { callable_(); }; std::function<void()> callable_; } CONCAT(CONCAT(OnExit_, __LINE__), _instance) = std::function<void()>([&]() -> void __VA_ARGS__)
