"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Any, Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplication operation.

    Args:
    ----
          x: A floating point number.
          y: A floating point number.

    Returns:
    -------
          Product of x * y.

    """
    return x * y


def id(x: float) -> float:
    """Identity operation.

    Args:
    ----
        x: A floating point number

    Returns:
    -------
        The same value as input unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Addition operation.

    Args:
    ----
        x: A floating point number.
        y: A floating point number

    Returns:
    -------
        The sum of x + y.

    """
    return x + y


def neg(x: float) -> float:
    """Negation operation.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Less than inequality sign.

    Args:
    ----
        x: A floating point number.
        y: A floating point number

    Returns:
    -------
        Truthness of x < y.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Equality sign.

    Args:
    ----
        x: A floating point number.
        y: A floating point number

    Returns:
    -------
        Truthness of x == y.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Max operation.

    Args:
    ----
        x: A floating point number.
        y: A floating point number

    Returns:
    -------
        Maximum of the two x,y.

    """
    return x if x >= y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value

    Args:
    ----
        x: A floating point number.
        y: A floating point number

    Returns:
    -------
        True if the absolute difference between x and y is less than eps, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        Sigmoid function applied on x.

    """
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (math.exp(x) + 1)


def relu(x: float) -> float:
    """Applies the ReLU activation functionRelu operation.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        Relu activation applied on x.

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        The expenential function of x.

    """
    return float(math.exp(x))


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x: A floating point number.

    Returns:
    -------
        The reciprocal of x.

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: A floating point number.
        y: A floating point number.

    Returns:
    -------
        Derivative of log times a second arg.

    """
    return inv(x) * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x: A floating point number.
        y: A floating point number.

    Returns:
    -------
        Derivative of reciprocal times a second arg.

    """
    return -(1 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: A floating point number.
        y: A floating point number.

    Returns:
    -------
        Derivative of relu times a second arg.

    """
    return 1 * y if x > 0 else 0


# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[..., Any], *my_list: Iterable[Any]) -> Iterable[Any]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        func: A callable function.
        my_list: A list of elements

    Returns:
    -------
        list with elements in my_list transformed by func.

    """
    for args in zip(*my_list):
        yield func(*args)


def zipWith(
    func: Callable[[Any, Any], Any], my_list1: Iterable[Any], my_list2: Iterable[Any]
) -> Iterable[Any]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
       func: A callable function.
       my_list1: A list of elements
       my_list2: A list of elements

    Returns:
    -------
       list with elements in my_list1 and my_list2 transformed by func.

    """
    return [func(a, b) for (a, b) in zip(my_list1, my_list2)]


def reduce(func: Callable[[Any, Any], float], my_list: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
      func: A callable function.
      my_list: A list of elements

    Returns:
    -------
      A single value obtained by applying func cumulatively to the items of my_list.

    """
    iterator = iter(my_list)
    try:
        result = next(iterator)
    except StopIteration:
        # raise ValueError("Cannot reduce an empty iterable")
        return 0

    for item in iterator:
        result = func(result, item)

    return result


def negList(my_list: Iterable[Any]) -> Iterable[Any]:
    """Negate all elements in a list using map

    Args:
    ----
      my_list: A list of elements

    Returns:
    -------
      A list of negated elements.

    """
    return map(neg, my_list)


def addLists(my_list1: Iterable[Any], my_list2: Iterable[Any]) -> Iterable[Any]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
      my_list1: A list of elements
      my_list2: A list of elements

    Returns:
    -------
      A list of summed up elements from both lists.

    """
    return zipWith(add, my_list1, my_list2)


def sum(my_list: Iterable[Any]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
      my_list: A list of elements

    Returns:
    -------
      A list of summed up elements.

    """
    return reduce(add, my_list)


def prod(my_list: Iterable[Any]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
      my_list: A list of elements

    Returns:
    -------
      The product of all elements.

    """
    return reduce(mul, my_list)
