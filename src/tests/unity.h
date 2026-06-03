/* Unity — Minimal C Test Framework (single-header, inspired by ThrowTheSwitch/Unity)
 * Stripped down for AGI project — just what we need, nothing more. */

#ifndef UNITY_H
#define UNITY_H

#include <math.h>
#include <stdio.h>
#include <string.h>

static int _unity_tests_run = 0;
static int _unity_tests_failed = 0;
static int _unity_tests_passed = 0;
static const char *_unity_current_test = NULL;

#define TEST_BEGIN(name)                                                                           \
    do {                                                                                           \
        _unity_current_test = #name;                                                               \
        _unity_tests_run++;                                                                        \
        printf("  TEST %-50s ", #name);                                                            \
    } while (0)

#define TEST_END()                                                                                 \
    do {                                                                                           \
        _unity_tests_passed++;                                                                     \
        printf("PASS\n");                                                                          \
    } while (0)

#define TEST_FAIL(msg)                                                                             \
    do {                                                                                           \
        _unity_tests_failed++;                                                                     \
        printf("FAIL\n");                                                                          \
        printf("    %s:%d: %s\n", __FILE__, __LINE__, msg);                                        \
        return;                                                                                    \
    } while (0)

#define ASSERT_TRUE(condition)                                                                     \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            TEST_FAIL("ASSERT_TRUE(" #condition ") failed");                                       \
        }                                                                                          \
    } while (0)

#define ASSERT_FALSE(condition)                                                                    \
    do {                                                                                           \
        if ((condition)) {                                                                         \
            TEST_FAIL("ASSERT_FALSE(" #condition ") failed");                                      \
        }                                                                                          \
    } while (0)

#define ASSERT_EQUAL_INT(expected, actual)                                                          \
    do {                                                                                           \
        long long _e = (long long)(expected);                                                      \
        long long _a = (long long)(actual);                                                        \
        if (_e != _a) {                                                                            \
            char _buf[256];                                                                        \
            snprintf(_buf, sizeof(_buf), "Expected %lld, got %lld", _e, _a);                       \
            TEST_FAIL(_buf);                                                                       \
        }                                                                                          \
    } while (0)

#define ASSERT_EQUAL_FLOAT(expected, actual, tolerance)                                             \
    do {                                                                                           \
        double _e = (double)(expected);                                                            \
        double _a = (double)(actual);                                                              \
        if (fabs(_e - _a) > (double)(tolerance)) {                                                 \
            char _buf[256];                                                                        \
            snprintf(_buf, sizeof(_buf), "Expected %.8f, got %.8f (tol %.8f)", _e, _a,             \
                     (double)(tolerance));                                                          \
            TEST_FAIL(_buf);                                                                       \
        }                                                                                          \
    } while (0)

#define ASSERT_EQUAL_STR(expected, actual)                                                          \
    do {                                                                                           \
        if (strcmp((expected), (actual)) != 0) {                                                    \
            char _buf[512];                                                                        \
            snprintf(_buf, sizeof(_buf), "Expected \"%s\", got \"%s\"", (expected), (actual));      \
            TEST_FAIL(_buf);                                                                       \
        }                                                                                          \
    } while (0)

#define ASSERT_NULL(ptr)                                                                           \
    do {                                                                                           \
        if ((ptr) != NULL) {                                                                       \
            TEST_FAIL("ASSERT_NULL(" #ptr ") failed — pointer is not NULL");                       \
        }                                                                                          \
    } while (0)

#define ASSERT_NOT_NULL(ptr)                                                                       \
    do {                                                                                           \
        if ((ptr) == NULL) {                                                                       \
            TEST_FAIL("ASSERT_NOT_NULL(" #ptr ") failed — pointer is NULL");                       \
        }                                                                                          \
    } while (0)

/* Array comparison for float buffers (critical for tensor ops) */
#define ASSERT_ARRAY_FLOAT(expected, actual, length, tolerance)                                     \
    do {                                                                                           \
        for (int _i = 0; _i < (int)(length); _i++) {                                               \
            if (fabs((double)(expected)[_i] - (double)(actual)[_i]) > (double)(tolerance)) {        \
                char _buf[256];                                                                    \
                snprintf(_buf, sizeof(_buf), "Array mismatch at index %d: expected %.8f, got %.8f", \
                         _i, (double)(expected)[_i], (double)(actual)[_i]);                        \
                TEST_FAIL(_buf);                                                                   \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define RUN_TEST(test_fn)                                                                          \
    do {                                                                                           \
        TEST_BEGIN(test_fn);                                                                       \
        test_fn();                                                                                 \
        if (_unity_tests_run > _unity_tests_passed + _unity_tests_failed) {                        \
            TEST_END();                                                                            \
        }                                                                                          \
    } while (0)

#define TEST_REPORT()                                                                              \
    do {                                                                                           \
        printf("\n========================================\n");                                     \
        printf("  %d tests, %d passed, %d failed\n", _unity_tests_run, _unity_tests_passed,       \
               _unity_tests_failed);                                                               \
        printf("========================================\n");                                       \
        return _unity_tests_failed > 0 ? 1 : 0;                                                   \
    } while (0)

#endif /* UNITY_H */
