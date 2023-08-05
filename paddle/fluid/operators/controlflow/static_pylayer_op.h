#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"



namespace paddle {
namespace operators {

class StaticPyLayerOp : public framework::OperatorBase {
    public:
        StaticPyLayerOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
                    : framework::OperatorBase(type, inputs, outputs, attrs) {}

    static const char kInputs[];
    static const char kOutputs[];
    static const char kScope[];
    static const char kSkipEagerDeletionVars[];

    private:
        void RunImpl(const framework::Scope & scope,
                    const platform::Place &dev_place) const override;

    private:
        mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};
};


class StaticPyLayerOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(StaticPyLayerOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(StaticPyLayerOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    // TODO: Use std::vector here ? 
    AddOutput(StaticPyLayerOp::kScope,
              "(std::vector<Scope*>) The scope of static pylayer block.");
    AddAttr<framework::BlockDesc *>(
        "sub_block", "The step block of conditional block operator");
    AddComment(R"DOC(StaticPyLayer operator

TO-DO: added by luqi


)DOC");
  }
};


}   // namespace operators
}   // namespace paddle