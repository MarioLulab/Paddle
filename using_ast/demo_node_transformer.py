import ast
import astunparse

func_def = \
"""
def add(x, y):
    return x + y
ret = add(3, 5)
print("hello world")
"""

r_node = ast.parse(func_def)

class CodeVisitor(ast.NodeVisitor):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        print('Function Name:{}'.format(node.name))
        self.generic_visit(node)
        # func_log_stmt = ast.Print(
        #     dest = None,
        #     values = [ast.Str(s = 'calling func: %s' % node.name, lineno = 0, col_offset = 0)],
        #     nl = True,
        #     lineno = 0,
        #     col_offset = 0,
        # )
        # func_log_stmt = ast.Call(
        #     func=ast.Name(
        #     id='print',
        #     ctx=ast.Load()),
        #     args=[ast.Str(s = 'calling func: {}'.format(node.name), lineno = 0, col_offset = 0)],
        #     keywords=[]
        # )
        # node.body.insert(0, func_log_stmt)

class CodeTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub
        self.generic_visit(node)
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        if node.name == 'add':
            node.name == 'sub'
        args_num = len(node.args.args)
        # args = tuple([arg.id for arg in node.args.args])
        args = tuple([arg for arg in node.args.args])
        # func_log_stmt = ''.join(["print 'calling func: {}', ".format(node.name), "'args:'", ", {}".format(args) * args_num])
        # node.body.insert(0, ast.parse(func_log_stmt))
        return node
    
    def visit_Name(self, node):
        replace = {
            "add": "sub",
            "x": "a",
            "y": "b"
        }
        print(f"node.id = {node.id}")
        re_id = replace.get(node.id, None)
        node.id = re_id if re_id else node.id
        self.generic_visit(node)
        return node

r_node = ast.parse(func_def)
# visitor = CodeVisitor()
# visitor.visit(r_node)
transformer = CodeTransformer()
r_node = transformer.visit(r_node)

with open("ast.log", "w") as f:    
    #print(ast.dump(r_node), file=f) 
    print(astunparse.dump(r_node), file=f)  # with good format 
print(astunparse.unparse(r_node))

exec(astunparse.unparse(r_node))