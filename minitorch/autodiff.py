from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus = list(vals)
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    visited = {variable.unique_id: False}
    results: List[Variable] = []
    __visit(variable, visited, results)
    results.reverse()
    return results


def __visit(
    variable: Variable, visited: dict[int, bool], results: list[Variable]
) -> None:
    if visited[variable.unique_id]:
        return

    for parent_node in variable.parents:
        if parent_node.unique_id not in visited:
            visited[parent_node.unique_id] = False
        __visit(parent_node, visited, results)

    visited[variable.unique_id] = True
    results.append(variable)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    scalar_derivative_mapping = defaultdict(lambda: 0)
    scalar_derivative_mapping[
        variable.unique_id
    ] = deriv  # dvar/dvar = 1, the derivative of root node is always 1
    sorted_nodes = topological_sort(variable)
    for curr_node in sorted_nodes:
        d_out = scalar_derivative_mapping[curr_node.unique_id]
        if curr_node.is_leaf():
            curr_node.accumulate_derivative(
                scalar_derivative_mapping[curr_node.unique_id]
            )
        else:
            scalars = curr_node.chain_rule(d_out)
            for (scalar, derivative_from_backward) in scalars:
                scalar_derivative_mapping[scalar.unique_id] += derivative_from_backward


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
