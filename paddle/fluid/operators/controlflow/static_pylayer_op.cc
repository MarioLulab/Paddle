#include "paddle/fluid/operators/controlflow/static_pylayer_op.h"

#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

const char StaticPyLayerOp::kInputs[] = "Input";
const char StaticPyLayerOp::kOutputs[] = "Out";
const char StaticPyLayerOp::kScope[] = "Scope";
const char StaticPyLayerOp::kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

class StaticPyLayerForwardOp : public StaticPyLayerOp {
    public:
        StaticPyLayerForwardOp(const std::string &type,
                            const framework::VariableNameMap &inputs,
                            const framework::VariableNameMap &outputs,
                            const framework::AttributeMap &attrs)
            : StaticPyLayerOp(type, inputs, outputs, attrs) {}
    private:
        void RunImpl(const framework::Scope & scope,
                const platform::Place &dev_place) const{
            // do nothing
            auto *scope_var = scope.FindVar(Output(kScope));
            PADDLE_ENFORCE_NOT_NULL(
            scope_var,
            platform::errors::PreconditionNotMet(
                "Expect Scope variable to be set in static_pylayer_op, but "
                "got a null Scope variable. Please set the Scope variable."));

            auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
            scopes->resize(1);
            scopes->front() = &scope.NewScope();

            auto &cur_scope = *scopes->front();
            // auto *block = Attr<framework::BlockDesc *>("sub_block");
            auto *block = Attr<framework::BlockDesc *>("forward_block");
            VLOG(3) << "Conditional block.idx = " << block->ID()
                    << ", scope = " << &cur_scope;

            auto &skip_vars =
                Attr<std::vector<std::string>>(kSkipEagerDeletionVars);

            LOG_FIRST_N(INFO, 1)
                << "[ControlFlow][StaticPyLayer] New Executor is Running.";

            if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
                VLOG(10) << "[interpreterCore cache]" << core_.get();
                VLOG_IF(10, core_) << platform::is_same_place(core_->GetPlace(),
                                                            dev_place);

                framework::interpreter::ExecutionConfig execution_config;
                execution_config.create_local_scope = false;
                execution_config.used_for_control_flow_op = true;
                execution_config.skip_gc_vars =
                    std::set<std::string>(skip_vars.begin(), skip_vars.end());

                core_.reset(new framework::InterpreterCore(
                    dev_place, *block, &cur_scope, execution_config));
                VLOG(10) << "[interpreterCore] created:" << core_;
            } else {
                // TODO: Add StaticPyLayer Helper ?
                BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
                core_->reset_scope(&cur_scope);
            }

            core_->Run({}, false);
        }

    private:
      mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};
};

class StaticPyLayerForwardInferShape : public framework::InferShapeBase {
    public:
        void operator()(framework::InferShapeContext *context) const override {
            // do nothing
        }
};

class StaticPyLayerBackwardOp : public StaticPyLayerOp {
    public:
        StaticPyLayerBackwardOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
                    : StaticPyLayerOp(type, inputs, outputs, attrs) {}
    
    private:
        void RunImpl(const framework::Scope & scope,
                    const platform::Place &dev_place) const override {
            const auto &inputs = Inputs(StaticPyLayerOp::kInputs);      //  "X"
            const auto &outside_grads =
                Outputs(framework::GradVarName(StaticPyLayerOp::kInputs));      // "X@GRAD"
            std::vector<std::string> inside_grads;
            inside_grads.reserve(inputs.size());
            for (auto &in : inputs) {
                inside_grads.emplace_back(framework::GradVarName(in));
            }

            // for debug
            std::cout << "==============" << std::endl;
            std::cout << "inside_grads = " << std::endl;
            for (auto& item : inside_grads) {
                std::cout << item << std::endl;
            }

            auto *scope_var = scope.FindVar(Input(StaticPyLayerOp::kScope));
            PADDLE_ENFORCE_NOT_NULL(
                scope_var,
                platform::errors::PreconditionNotMet(
                    "Expect Scope variable to be set in conditional_block_op, but "
                    "got a null Scope variable. Please set the Scope variable."));
            auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
            PADDLE_ENFORCE_GT(
                scopes.size(),
                0,
                platform::errors::InvalidArgument(
                    "Expect Scope variable contains at least 1 scope, but got: %d",
                    scopes.size()));
            framework::Scope &cur_scope = *(scopes[0]);

            // auto *block = Attr<framework::BlockDesc *>("sub_block");
            auto *block = Attr<framework::BlockDesc *>("forward_block");
            VLOG(3) << "Static PyLayer Grad block.idx = " << block->ID()
                    << ", scope = " << &cur_scope;

            LOG_FIRST_N(INFO, 1)
                << "[ControlFlow][ConditionalGradBlock] New Executor is Running.";
            if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
                VLOG(10) << "[interpreterCore cache]" << core_.get();
                VLOG_IF(10, core_) << platform::is_same_place(core_->GetPlace(),
                                                            dev_place);

                framework::interpreter::ExecutionConfig execution_config;
                execution_config.create_local_scope = false;
                execution_config.used_for_control_flow_op = true;
                execution_config.skip_gc_vars =
                    std::set<std::string>(inside_grads.begin(), inside_grads.end());

                core_.reset(new framework::InterpreterCore(
                    dev_place, *block, &cur_scope, execution_config));
                VLOG(10) << "[interpreterCore] created:" << core_;
            } else {
                BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
                core_->reset_scope(&cur_scope);
            }
            core_->Run({}, false);

            // necessary ?
            AssignLocalGradientToParentScope(
                dev_place, cur_scope, scope, inside_grads, outside_grads, inputs);
            // Release the cur_scope, otherwise memory leakage occurs.
            scope.DeleteScope(&cur_scope);
            return;
    }

    private:
        mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};

    private:
      void AssignLocalGradientToParentScope(
      const platform::Place &place,
      const framework::Scope &cur_scope,
      const framework::Scope &parent_scope,
      const std::vector<std::string> &inside_grads,
      const std::vector<std::string> &outside_grads,
      const std::vector<std::string> &inputs) const {
        std::vector<std::string> assign_zero_outside_grads;
        std::vector<std::string> assign_zero_inputs;
        for (size_t i = 0; i < outside_grads.size(); ++i) {
        const std::string &outside_grad_name = outside_grads[i];
        const std::string &inside_grad_name = inside_grads[i];
        VLOG(4) << "[assign local]"
                << "inside_grad_name = " << inside_grad_name
                << ", outside_grad_name = " << outside_grad_name;
        framework::Variable *outside_var =
            parent_scope.FindVar(outside_grad_name);
        if (outside_var == nullptr) {
            continue;
        }
        framework::Variable *inside_var =
            cur_scope.FindLocalVar(inside_grad_name);
        if (inside_var == nullptr) {
            assign_zero_outside_grads.emplace_back(outside_grad_name);
            assign_zero_inputs.emplace_back(inputs[i]);
            continue;
        }
        platform::DeviceContext *dev_ctx =
            platform::DeviceContextPool::Instance().Get(place);
        framework::VisitVarType(*inside_var,
                                AssignFunctor(outside_var, *dev_ctx));
        }
    }

};

class StaticPyLayerBackwardInferShape : public framework::InferShapeBase {
    public:
        void operator()(framework::InferShapeContext *context) const override {
            if (context->HasInputs(StaticPyLayerOp::kInputs) &&
                context->HasOutputs(framework::GradVarName(StaticPyLayerOp::kInputs))) {
                    context->SetOutputsDim(framework::GradVarName(StaticPyLayerOp::kInputs),
                                    context->GetInputsDim(StaticPyLayerOp::kInputs));
                }
        }
};


class StaticPyLayerBackwardInferVarType : public framework::VarTypeInference {
    public:
        void operator()(framework::InferVarTypeContext *ctx) const override {
            auto input_size = ctx->InputSize(StaticPyLayerOp::kInputs);
            auto output_size =
                ctx->OutputSize(framework::GradVarName(StaticPyLayerOp::kInputs));
            PADDLE_ENFORCE_EQ(input_size,
                            output_size,
                            platform::errors::InvalidArgument(
                                "input_size and output_size should be equal for "
                                "static_pylayer_grad_op."));
            for (size_t i = 0; i < output_size; ++i) {
                ctx->SyncTypeAndDataType(StaticPyLayerOp::kInputs,
                                        framework::GradVarName(StaticPyLayerOp::kInputs),
                                        i);
            }
        }
};

template <class T>
struct FilterNoGradInput {};

template <>
struct FilterNoGradInput<framework::OpDesc> {
  static void filter(const framework::BlockDesc *desc,
                     std::vector<std::string> *vec) {
    auto f = [desc](const std::string &name) -> std::string {
      if (name == framework::kEmptyVarName) {
        // don't drop empty var name, you can use Input(name, true) to drop
        // it.
        return framework::kEmptyVarName;
      }
      auto var_desc =
          desc->FindVarRecursive(framework::GradOriginalVarName(name));
      std::set<framework::proto::VarType::Type> not_support_backward_dtype = {
          framework::proto::VarType::BOOL,
          framework::proto::VarType::INT8,
          framework::proto::VarType::UINT8,
          framework::proto::VarType::INT16,
          framework::proto::VarType::INT32,
          framework::proto::VarType::INT64,
      };
      if (!var_desc ||
          not_support_backward_dtype.count(var_desc->GetDataType()))
        return framework::kEmptyVarName;
      return name;
    };
    std::transform(vec->begin(), vec->end(), vec->begin(), f);
  }
};

template <typename T>
class StaticPyLayerBackwardMaker : public framework::SingleGradOpMaker<T> {
    public:
        using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

    protected:
        void Apply(GradOpPtr<T> grad_op) const override {
            grad_op->SetType("static_pylayer_grad");
            // NOTE: Just For TypeInfer in GradOp
            grad_op->SetInput(StaticPyLayerOp::kInputs,
                            this->Input(StaticPyLayerOp::kInputs));
            grad_op->SetInput(framework::GradVarName(StaticPyLayerOp::kOutputs),
                              this->OutputGrad(StaticPyLayerOp::kOutputs)); // second is `var` name
            grad_op->SetInput(StaticPyLayerOp::kScope,
                            this->Output(StaticPyLayerOp::kScope));

            auto fwd_inputs = this->InputGrad(StaticPyLayerOp::kInputs, false);
            // NOTE: no need to filt
            // FilterNoGradInput<T>::filter(this->GetForwardOpBlock(), &fwd_inputs);
            grad_op->SetOutput(framework::GradVarName(StaticPyLayerOp::kInputs),
                            fwd_inputs);
            // grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
            // grad_op->SetBlockAttr("sub_block", this->Attr<framework::BlockDesc *>("backward_block"));
            // grad_op->SetBlockAttr("sub_block", PADDLE_GET_CONST(framework::BlockDesc *, this->GetAttr("backward_block")));
            grad_op->SetBlockAttr("forward_block", PADDLE_GET_CONST(framework::BlockDesc *, this->GetAttr("backward_block")));
        }

};

}   // namespace operators
}   // namespace


namespace ops = paddle::operators;
REGISTER_OPERATOR(static_pylayer,
                  ops::StaticPyLayerForwardOp,
                  ops::StaticPyLayerForwardInferShape,
                  ops::StaticPyLayerForwardOpProtoMaker,
                  ops::StaticPyLayerBackwardMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(static_pylayer_grad,
                  ops::StaticPyLayerBackwardOp,
                  ops::StaticPyLayerBackwardInferShape,
                  ops::StaticPyLayerBackwardInferVarType);
