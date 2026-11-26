// Unit tests for objc_method_hooks: a method replaced on classes made at run
// time, so no Metal object is touched. No GPU.

#include "objc_method_hooks.h"
#include "test_harness.h"

#include <objc/runtime.h>

namespace {

using AnswerImp = int (*)(id, SEL);

int answer_forty(id, SEL) { return 40; }

AnswerImp original_answer = nullptr;

// Calls the implementation it replaced, as the recorders' hooks do.
int answer_hook(id receiver, SEL selector) { return original_answer(receiver, selector) + 2; }

Class make_class(const char* class_name, Class superclass, bool has_own_answer) {
  Class made_class = objc_getClass(class_name);
  if (made_class != nullptr) return made_class;
  made_class = objc_allocateClassPair(superclass, class_name, 0);
  if (has_own_answer) {
    class_addMethod(made_class, sel_registerName("answer"), reinterpret_cast<IMP>(answer_forty), "i@:");
  }
  objc_registerClassPair(made_class);
  return made_class;
}

// The method a message to an instance of `target_class` would run, called
// directly (no instance is needed: the implementations ignore `self`).
int call_answer(Class target_class) {
  const SEL selector = sel_registerName("answer");
  return reinterpret_cast<AnswerImp>(class_getMethodImplementation(target_class, selector))(nullptr, selector);
}

// --- positive ---

void should_run_the_hook_which_reaches_the_original() {
  const Class target_class = make_class("MlxPerfHookTargetOwn", objc_getClass("NSObject"), true);
  EXPECT_TRUE(mlx_perf::replace_method(target_class, "answer", reinterpret_cast<IMP>(answer_hook),
                                       original_answer));
  EXPECT_TRUE(original_answer == answer_forty);
  EXPECT_TRUE(call_answer(target_class) == 42);
}

// --- edge / boundary ---

void should_hook_an_inherited_method_on_the_subclass_only() {
  const Class base_class = make_class("MlxPerfHookBase", objc_getClass("NSObject"), true);
  const Class derived_class = make_class("MlxPerfHookDerived", base_class, false);
  original_answer = nullptr;
  EXPECT_TRUE(mlx_perf::replace_method(derived_class, "answer", reinterpret_cast<IMP>(answer_hook),
                                       original_answer));
  EXPECT_TRUE(original_answer == answer_forty);
  EXPECT_TRUE(call_answer(derived_class) == 42);
  EXPECT_TRUE(call_answer(base_class) == 40);
}

// --- negative ---

void should_refuse_a_selector_the_class_does_not_have_and_change_nothing() {
  const Class target_class = make_class("MlxPerfHookTargetBare", objc_getClass("NSObject"), false);
  AnswerImp untouched_original = nullptr;
  EXPECT_TRUE(!mlx_perf::replace_method(target_class, "noSuchSelectorForMlxPerf",
                                        reinterpret_cast<IMP>(answer_hook), untouched_original));
  EXPECT_TRUE(untouched_original == nullptr);
  EXPECT_TRUE(class_getInstanceMethod(target_class, sel_registerName("noSuchSelectorForMlxPerf")) == nullptr);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_run_the_hook_which_reaches_the_original),
      TEST_CASE(should_hook_an_inherited_method_on_the_subclass_only),
      TEST_CASE(should_refuse_a_selector_the_class_does_not_have_and_change_nothing),
  });
}
