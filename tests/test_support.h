#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline void require(const bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

inline void near(const double actual, const double expected, const double tolerance = 1e-10)
{
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected)))
        throw std::runtime_error("attendu=" + std::to_string(expected)
            + ", obtenu=" + std::to_string(actual));
}

template <class Exception, class Callable>
void expect_throw(Callable callable)
{
    bool thrown = false;
    try { callable(); }
    catch (const Exception&) { thrown = true; }
    require(thrown, "exception attendue absente");
}

inline int run_tests(const std::vector<std::pair<std::string, std::function<void()>>>& tests)
{
    int failed = 0;
    for (const auto& [name,test] : tests)
    {
        try
        {
            test();
            std::cout << "PASS " << name << '\n';
        }
        catch (const std::exception& e)
        {
            ++failed;
            std::cout << "FAIL " << name << " : " << e.what() << '\n';
        }
    }
    std::cout << "TOTAL " << tests.size() << " | PASS " << tests.size() - failed
        << " | FAIL " << failed << '\n';
    return failed ? 1 : 0;
}
