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


class StaticPyLayerBlockGuard(BlockGuard):
    def __init__(self, block):
        check_type(block, "block", StaticPyLayerBlock)
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
    
    def block(self):
        return StaticPyLayerBlockGuard(self)
    
    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

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

        static_pylayer_op = parent_block.append(
            type='static_pylayer',
            inputs={
                'Input': param_list,
            },
            outputs={
                "Out": out_list,
            },
            attrs={
                'sub_block': inside_block,
            }
        )