// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::dialect::PyLayerOp
#else

#include <unordered_map>

#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace paddle {
namespace dialect {

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      pir::Value combined_inputs,
                      std::vector<pir::Type> &&output_types) {
  argument.AddInput(combined_inputs);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
}

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      pir::Value combined_inputs,
                      std::unique_ptr<pir::Block> &&fwd_block) {
  VLOG(4) << "Start build PyLayerOp";

  PADDLE_ENFORCE_NOT_NULL(fwd_block,
                          paddle::platform::errors::InvalidArgument(
                              "The sub-block for building pylayer_op "
                              "can't be None"));

  PADDLE_ENFORCE_NE(fwd_block->empty(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The sub-block for building pylayer_op "
                        "can't be empty"));

  PADDLE_ENFORCE_EQ(fwd_block->back().isa<pir::YieldOp>(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The last op of sub-block for building pylayer_op "
                        "must be pir::YieldOp"));

  auto &op = fwd_block->back();

  std::vector<pir::Attribute> outs_stop_gradient;
  for (size_t i = 0; i < op.num_operands(); ++i) {
    argument.AddOutput(op.operand(i).type());
    auto bool_attr = op.operand_source(i).attribute<pir::BoolAttribute>(
        pir::kStopGradientAttrName);
    outs_stop_gradient.push_back(bool_attr ? bool_attr
                                           : builder.bool_attr(false));
  }

  argument.AddAttribute(
      pir::kStopGradientAttrName,
      pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));

  argument.AddRegion().push_back(fwd_block.release());
  argument.AddInput(combined_inputs);
}

pir::Block &PyLayerOp::forward_block() {
  pir::Region &region = forward_region();
  if (region.empty()) {
    region.emplace_back();
  }

  return region.front();
}

void PyLayerOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = pd_op.pylayer";
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << "{";
  for (auto &item : forward_block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n }";
}

void PyLayerOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: PyLayerOp.";
  // NOTE(MarioLulab): do nothing.
}

void PyLayerOp::VerifyRegion() {
  VLOG(4) << "Start Verifying sub regions for: PyLayerOp.";
  VLOG(4) << "Start Verifying forward block.";
  PADDLE_ENFORCE_EQ((*this)->region(0).size(),
                    1u,
                    phi::errors::PreconditionNotMet(
                        "The size %d of forward_region must be 1.",
                        (*this)->region(0).size()));
  if ((*this)->num_results() != 0) {
    auto &fwd_last_op = (*this)->region(0).front().back();
    PADDLE_ENFORCE_EQ(true,
                      fwd_last_op.isa<pir::YieldOp>(),
                      phi::errors::PreconditionNotMet(
                          "The last of forward block must be YieldOp"));
    PADDLE_ENFORCE_EQ(
        fwd_last_op.num_operands(),
        (*this)->num_results(),
        phi::errors::PreconditionNotMet(
            "The size of last of forward block op's input must be "
            "equal to PyLayerOp's outputs num."));
  }
}

void PyLayerOp::UpdateOutput() {
  PADDLE_ENFORCE_NOT_NULL(*this,
                          paddle::platform::errors::InvalidArgument(
                              "The pylayer_op in PyLayerOp used to update "
                              "output can't be nullptr"));
  auto block = parent();
  PADDLE_ENFORCE_NOT_NULL(
      block,
      paddle::platform::errors::InvalidArgument(
          "The parent block of pylayer_op which used to update "
          "output can't be nullptr"));
  pir::Block::Iterator iter = **this;
  pir::Builder builder(ir_context(), false);
  auto new_pylayer_op =
      builder.Build<PyLayerOp>(combined_inputs(), forward_region().TakeBack());
  block->Assign(iter, new_pylayer_op);
  PyLayerOp::operator=(new_pylayer_op);
  VerifyRegion();
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PyLayerOp)

#endif
