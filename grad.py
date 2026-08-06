class GradScheduler:
    def __init__(self):
        self.events = []

    def schedule(self, grad_fn, grad_val):
        self.events.append((grad_fn, grad_val))

    def execute(self):
        while self.events:
            grad_fn, grad_val = self.events.pop(0)
            grad_fn.backward(grad_val)

scheduler = GradScheduler()

class GradNode:
    def __init__(self):
        self.children_count = 0
        self.parents = None

    def backward(self, grad_output):
        raise NotImplementedError("Subclasses must implement the backward method.")

    def schedule(self, grad_val):
        self.children_count -= 1
        if self.children_count <= 0:
            scheduler.schedule(self, grad_val)

class AddBackward(GradNode):
    def __init__(self, a, b):
        super().__init__()
        self.parents = a, b

    def backward(self, grad_output):
        a, b = self.parents
        a.grad_val += grad_output
        b.grad_val += grad_output
        if a.grad_node is not None:
            a.grad_node.schedule(a.grad_val)
        if b.grad_node is not None:
            b.grad_node.schedule(b.grad_val)

class MulBackward(GradNode):
    def __init__(self, a, b):
        super().__init__()
        self.parents = a, b

    def backward(self, grad_output):
        a, b = self.parents
        a.grad_val += grad_output * b.data
        b.grad_val += grad_output * a.data
        if a.grad_node is not None:
            a.grad_node.schedule(a.grad_val)
        if b.grad_node is not None:
            b.grad_node.schedule(b.grad_val)

class Tensor:
    def __init__(self, value):
        self.grad_node = None
        self.data = value
        self.grad_val = 0

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data)
            result.grad_node = AddBackward(self, other)
            return result
        else:
            raise ValueError("Operand must be a Tensor.")

    def __mul__(self, other):
        if isinstance(other, Tensor):

            result = Tensor(self.data * other.data)
            result.grad_node = MulBackward(self, other)
            return result
        else:
            raise ValueError("Operand must be a Tensor.")

    def backward(self):
        if self.grad_val == 0:
            self.grad_val = 1
        if self.grad_node is not None:
            backward_dfs(self.grad_node)
            self.grad_node.schedule(self.grad_val)


def zero_grad(root: Tensor):
    root.grad_val = 0
    if root.grad_node is not None:
        for parent in root.grad_node.parents:
            zero_grad(parent)

def backward_dfs(root_node: GradNode):
    # pass 1: collect every grad_node reachable from the root (once each)
    # and reset its dependency count for this backward pass
    visited = set()

    def collect(node):
        if node in visited:
            return
        visited.add(node)
        node.children_count = 0
        for parent in node.parents:
            if parent.grad_node is not None:
                collect(parent.grad_node)

    collect(root_node)

    # pass 2: count consumer edges — each node increments the counts of
    # its parents' grad_nodes. Duplicate parents (e.g. c + c) count twice,
    # matching the two contributions delivered when this node fires.
    for node in visited:
        for parent in node.parents:
            if parent.grad_node is not None:
                parent.grad_node.children_count += 1

a = Tensor(6)
b = Tensor(7)
c = a + b
e = Tensor(3)
f = c + e
g = f * c
g.backward()
scheduler.execute()
print(f'a grad: {a.grad_val}, b grad: {b.grad_val}')
assert a.grad_val == 29, f'a grad: {a.grad_val}, expected: 29'  # dg/da = f + c = 16 + 13

z = a + a
k = z * z
zero_grad(k)
k.backward()
scheduler.execute()
assert a.grad_val == 48, f'a grad: {a.grad_val}, expected: 48'