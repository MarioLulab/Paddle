from ..wrapped_decorator import signature_safe_contextmanager

from .layer_function_generator import templatedoc
from .. import core
from ..framework import (
    Program,
    Variable,
    Operator,
    static_only,
    in_dygraph_mode,
)
from ..layer_helper import LayerHelper, unique_name
from ...utils import (
    assert_same_structure,
    map_structure,
    hold_mutable_vars,
    copy_mutable_vars,
    is_sequence,
    pack_sequence_as,
    flatten,
    to_sequence,
)
import numpy
import warnings
from functools import reduce, partial
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from ..backward import _infer_var_data_type_shape_
import paddle
from paddle import _C_ops, _legacy_C_ops

from .control_flow import BlockGuard

__all__ = [
    'StaticPyLayerBlock',
]

# NOTE: copy from control flow...
# (TODO: Mine) There exists dependency. It will be removed later.
def get_inputs_outputs_in_block(
    current_block, inner_inputs, inner_outputs, helper
):
    """
    Find inputs and outputs in current control flow block.
    :param current_block: Current control flow block.
    :param inner_inputs: Input var name of ops in current block.
    :param inner_outputs: Output var name of ops in current block.
    :return: inner_inputs, inner_outputs
    """

    def is_ignore_vars(op, var_name):
        # NOTE(dev): There are some persistable var created in some non-standard API
        # such as "contrib.layers.shuffle_batch". It create a "Seed" used both in
        # Input and Output. This var shall not be considered as a loop_var in
        # control_flow.
        IGNORE_VAR_NAMES = {"shuffle_batch": ["shuffle_batch_seed"]}
        if op.type in IGNORE_VAR_NAMES:
            var_names = IGNORE_VAR_NAMES[op.type]
            for name in var_names:
                if name in var_name:
                    return True
        return False

    # Step1: update inner_inputs and inner_outputs
    # NOTE: Here assumes that all variables are input or output of Ops,
    # but some variables are created without appendding a real op.
    # For example, in `arr = create_array(dtype)`, `arr` is not a output of a op.
    for op in current_block.ops:
        assert isinstance(op, Operator)
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if in_var_name not in inner_outputs and not is_ignore_vars(
                    op, in_var_name
                ):
                    inner_inputs.add(in_var_name)

        for oname in op.output_names:
            for out_var_name in op.output(oname):
                inner_outputs.add(out_var_name)

    # Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.
    remove_inner_inputs = set()
    parent_block = helper.main_program.block(current_block.parent_idx)

    for in_var_name in inner_inputs:
        parent_block_var = parent_block._find_var_recursive(in_var_name)
        current_block_var = None
        if current_block.has_var(in_var_name):
            current_block_var = current_block.var(in_var_name)
        if (
            not parent_block_var
            and current_block_var
            and current_block_var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY
        ):
            remove_inner_inputs.add(in_var_name)

    inner_inputs = inner_inputs - remove_inner_inputs

    return inner_inputs, inner_outputs

class StaticPyLayerBlockGuard(BlockGuard):
    def __init__(self, block):
        check_type(block, "block", StaticPyLayerBlock, "StaticPyLayerBlockGuard")
        super().__init__(block.helper.main_program)
        self.block = block
    
    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super().__exit__(exc_type, exc_val, exc_tb)

class StaticPyLayerBlock:
    def __init__(self, inputs, name=None):
        for each_input in inputs:
            check_type(each_input, "input", Variable, "StaticPyLayerBlock")
        
        self.helper = LayerHelper("static_pylayer_block", name=name)
    
    def block(self, is_backward_block=False):
        self.is_backward_block = is_backward_block
        return StaticPyLayerBlockGuard(self)
    
    @property
    def inside_block_index(self):
        return self.block_id
    
    def complete(self):        
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        self.block_id = inside_block.idx
        
        if self.is_backward_block:
            # set OpRole to `backward`
            for op in inside_block.ops:
                op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
                backward = core.op_proto_and_checker_maker.OpRole.Backward
                op.desc._set_attr(op_role_attr_name, backward)
            
            # exit, because there is no need to append 'static_pylayer' op
            return

        intermediate = set()
        params = set()
        params, intermediate = get_inputs_outputs_in_block(
            inside_block, params, intermediate, helper=self.helper
        )

        param_list = [
            parent_block._var_recursive(each_name) for each_name in params
        ]

        out_list = []
        for inner_out_name in intermediate:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_list.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )
        static_pylayer_op = parent_block.append_op(
            type='static_pylayer',
            inputs={
                'Input': param_list,
            },
            outputs={
                "Out": out_list,
                "Scope": [step_scope]
            },
            attrs={
                'sub_block': inside_block,
            }
        )