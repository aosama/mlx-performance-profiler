#pragma once

// Objective-C method hooks: one concern only — swap one method of a class for
// a hook while keeping the implementation it replaces, so the hook can call
// it. Used by the recorders that observe MLX's Metal objects; each recorder
// owns its own hooks and what they record.

#include <objc/runtime.h>

namespace mlx_perf {

// Replaces `selector_name` on `target_class` with `hook_implementation` and
// stores the implementation it replaces (possibly inherited from a
// superclass) in `original_implementation`. The original is stored before
// the swap, so no call can reach a hook whose original is unset. False, with
// nothing changed, when the class does not respond to the selector.
template <typename ImplementationType>
bool replace_method(Class target_class, const char* selector_name, IMP hook_implementation,
                    ImplementationType& original_implementation) {
  const SEL selector = sel_registerName(selector_name);
  const Method current_method = class_getInstanceMethod(target_class, selector);
  if (current_method == nullptr) {
    return false;
  }
  original_implementation = reinterpret_cast<ImplementationType>(method_getImplementation(current_method));
  class_replaceMethod(target_class, selector, hook_implementation, method_getTypeEncoding(current_method));
  return true;
}

} // namespace mlx_perf
