#include "torch_xla/csrc/runtime/xrt_session.h"

#include "absl/strings/str_cat.h"

namespace torch_xla {
namespace runtime {

XrtSession::XrtSession(const tensorflow::SessionOptions& session_options)
    : target_(session_options.target),
      root_(tensorflow::Scope::NewRootScope()),
      session_(root_, session_options) {}

void XrtSession::Reset() {
  for (auto& name_cache : node_cache_) {
    name_cache.second.Rewind();
  }
}

std::string XrtSession::GetCacheKey(const std::string& op_name,
                                    const std::string& device) {
  return absl::StrCat(op_name, ";", device);
}

}  // namespace runtime
}  // namespace torch_xla
