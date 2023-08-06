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
from paddle.fluid.framework import Block, Operator, Program, Variable, static_only

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
import re

from .control_flow import copy_var_to_parent_block

def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given variable name
    e.g. x@GRAD ==> x
         x@GRAD@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
         z@GRAD_slice_0@GRAD ==> z@GRAD_slice_0
         grad/grad/z@GRAD@RENAME@block0@1@GRAD ==> z
    """
    pos = re.search(f'{core.grad_var_suffix()}+@', name) or re.search(
        f'{core.grad_var_suffix()}$', name
    )
    new_name = name[: pos.start()] if pos is not None else name
    new_pos = name.rfind('grad/')
    return new_name[new_pos + 5 :] if new_pos != -1 else new_name


def do_static_pylayer(forward_fn, inputs, backward_fn, name=None, return_names=None):

    if in_dygraph_mode():
        raise NotImplementedError()

    check_type(name, "name", (str, type(None)), "fluid.layers.static_pylayer")
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)

    # only support position args now
    assert forward_fn is not None and callable(forward_fn)
    assert isinstance(inputs, list)
    static_pylayer_block = StaticPyLayerBlock(inputs)
    with static_pylayer_block.block():
        origin_output = forward_fn(*inputs)
        if origin_output is not None:
            output = map_structure(
                copy_to_parent_func, origin_output
            )
        
    # copy 一份 `origin_output` or `output` 作为输入构建 backward block ?
    current_block = helper.main_program.current_block()
    static_pylayer_op = current_block.ops[-1]
    no_grad_dict = set()
    grad_op_descs, op_grad_to_var = core.get_grad_op_desc(
        static_pylayer_op.desc, no_grad_dict, [helper.main_program.desc.block(static_pylayer_block.block_id)]
    )
    grad_op_desc = grad_op_descs[0]
    grad_var_name_ins = [
        var_name for var_name in grad_op_desc.input_arg_names() if core.grad_var_suffix() in var_name
    ]
    grad_var_name_outs = [
        var_name for var_name in grad_op_desc.output_arg_names() if core.grad_var_suffix() in var_name
    ]
    
    grad_var_ins = []
    for arg in grad_var_name_ins:
        # do some judge
        # ....
        
        fwd_name = _strip_grad_suffix_(arg)
        var = current_block.create_var(name=arg)

        if current_block.desc.has_var_recursive(fwd_name.encode()):
            fwd_var = current_block.desc.find_var_recursive(fwd_name.encode())
            var.desc.set_dtype(fwd_var.dtype())
            var.desc.set_shape(fwd_var.shape())
        else:
            # TODO(jiabin): Maybe we should not to this to cause some unexpected error on dtype
            warnings.warn(
                "Set grad var: {} dtype to default FP32, since we can't find its related forward var".format(
                    arg
                )
            )
            var.set_dtype(core.VarDesc.VarType.FP32)
            
        grad_var_ins.append(var)
    
    
    assert backward_fn is not None and callable(backward_fn)
    assert isinstance(grad_var_ins, list)
    static_pylayer_backward_block = StaticPyLayerBlock(grad_var_ins)
    with static_pylayer_backward_block.block(is_backward_block=True):
        grad_output = backward_fn(*grad_var_ins)

    for arg in grad_var_name_ins:
        current_block._remove_var(arg)


    if output is None:
        return None

    return output
