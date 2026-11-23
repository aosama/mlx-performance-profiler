#pragma once

// A deliberately tiny test runner: named test functions, non-fatal
// expectations, a non-zero exit code on any failure (which CTest reads).
// Chosen over GoogleTest to keep Phase 0–2 free of fetched dependencies.

#include <cmath>
#include <cstdio>
#include <initializer_list>

namespace test_harness {

struct TestCase {
  const char* name;
  void (*run)();
};

inline int& failed_expectation_count() {
  static int failed_expectations = 0;
  return failed_expectations;
}

inline void record_failure(const char* file, int line, const char* expression) {
  ++failed_expectation_count();
  std::printf("    FAILED %s:%d  %s\n", file, line, expression);
}

inline int run_all(std::initializer_list<TestCase> test_cases) {
  int failed_test_count = 0;
  for (const TestCase& test_case : test_cases) {
    const int failures_before = failed_expectation_count();
    test_case.run();
    const bool has_passed = failed_expectation_count() == failures_before;
    failed_test_count += has_passed ? 0 : 1;
    std::printf("%s  %s\n", has_passed ? "PASS" : "FAIL", test_case.name);
  }
  std::printf(
      "%zu tests, %d failed\n", test_cases.size(), failed_test_count);
  return failed_test_count == 0 ? 0 : 1;
}

} // namespace test_harness

#define EXPECT_TRUE(condition)                                         \
  do {                                                                 \
    if (!(condition)) {                                                \
      test_harness::record_failure(__FILE__, __LINE__, #condition);   \
    }                                                                  \
  } while (false)

#define EXPECT_NEAR(actual, expected, tolerance)                       \
  EXPECT_TRUE(std::fabs((actual) - (expected)) <= (tolerance))

#define TEST_CASE(function_name) test_harness::TestCase{#function_name, function_name}
