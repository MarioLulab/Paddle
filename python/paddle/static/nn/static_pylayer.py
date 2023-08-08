import warnings
from functools import partial, reduce
import collections

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

# NOTE: Borrowed from `backward.py`
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


# NOTE: Borrowed from `backward.py`
def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    if isinstance(op_descs, (list, tuple)):
        for i in range(begin_idx, end_idx):
            op_desc = op_descs[i]
            if isinstance(op_desc, tuple):
                op_desc = op_desc[0]
            op_desc._rename_input(old_name, new_name)
            op_desc._rename_output(old_name, new_name)
    if isinstance(op_descs, collections.OrderedDict):
        for key, value in op_descs.items():
            if isinstance(value, (list, tuple)):
                for op_desc in value:
                    op_desc._rename_input(old_name, new_name)
                    op_desc._rename_output(old_name, new_name)

# NOTE: Borrowed from `backward.py`
def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return name + core.grad_var_suffix()



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
        
    # copy 一份 `origin_output` or `output` 作为输入构建 backward block, 随后删掉
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
    
    # push
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
    var_old_to_new = dict()
    with static_pylayer_backward_block.block(is_backward_block=True):
        grad_origin_output = backward_fn(*grad_var_ins)
        if grad_origin_output is not None:
            flat_grad_origin = flatten(grad_origin_output)
            forward_input_names = current_block.ops[static_pylayer_block.op_index].desc.input_arg_names()
            for idx, grad_out_old in enumerate(flat_grad_origin):
                # attach old var name into new
                forward_input_name = forward_input_names[idx]
                grad_out_new = _append_grad_suffix_(forward_input_name)
                var_old_to_new[grad_out_old.name] = grad_out_new

    # pop
    for arg in grad_var_name_ins:
        current_block._remove_var(arg)

    backward_block_desc = current_block.program.block(static_pylayer_backward_block.block_id).desc
    for old_var_name, new_var_name in var_old_to_new.items():
        backward_block_desc._rename_var(old_var_name.encode(), new_var_name.encode())
        for op_idx in range(backward_block_desc.op_size()):
            op = backward_block_desc.op(op_idx)
            op._rename_input(old_var_name, new_var_name)
            op._rename_output(old_var_name, new_var_name)

    # set 'backward_block' attr to forward op
    current_block.ops[static_pylayer_block.op_index].desc.set_block_attr("backward_block", backward_block_desc)

    if output is None:
        return None

    return output
