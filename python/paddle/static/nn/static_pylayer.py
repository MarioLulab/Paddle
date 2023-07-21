import warnings
from functools import partial, reduce

import paddle
from paddle.common_ops_import import (
    LayerHelper,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
    in_dygraph_mode,
)
from paddle.fluid import core
from paddle.fluid.framework import Operator, Program, Variable, static_only

# Temporary solution, it will be deleted later
# from paddle.fluid.layers.control_flow import ConditionalBlock, select_input
from paddle.fluid.layers.static_pylayer import StaticPyLayerBlock

from paddle.utils import (
    assert_same_structure,
    copy_mutable_vars,
    flatten,
    hold_mutable_vars,
    is_sequence,
    map_structure,
    pack_sequence_as,
    to_sequence,
)

from .control_flow import copy_var_to_parent_block

def do_static_pylayer(fn, inputs, name=None, return_names=None):

    if in_dygraph_mode():
        raise NotImplementedError()

    check_type(name, "name", (str, type(None)), "fluid.layers.static_pylayer")
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)

    # only support position args now
    assert fn is not None and callable(fn)
    assert isinstance(inputs, list)
    static_pylayer_block = StaticPyLayerBlock(inputs)
    with static_pylayer_block.block():
        origin_output = fn(*inputs)
        if origin_output is not None:
            output = map_structure(
                copy_to_parent_func, origin_output
            )
    
    if output is None:
        return None

    return output
