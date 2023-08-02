#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"



namespace paddle {
namespace operators {
static constexpr char kInputs[] = "Input";
static constexpr char kOutputs[] = "Out";
static constexpr char kScope[] = "Scope";
static constexpr char kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

class StaticPyLayerOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    // TODO: Use std::vector here ? 
    AddOutput(kScope,
              "(Scope*) The scope of static pylayer block.");
    AddAttr<framework::BlockDesc *>(
        "sub_block", "The step block of conditional block operator");
    AddComment(R"DOC(StaticPyLayer operator

TO-DO: added by luqi


)DOC");
  }
};


}   // namespace operators
}   // namespace paddle