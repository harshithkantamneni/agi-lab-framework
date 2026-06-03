#!/usr/bin/env python3
"""Symbolic Math Engine — Local computation for AGI research."""

import argparse
import json
import sys
import sympy
from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


# Make common symbols globally available
x, y, z, t, n, k, m = symbols("x y z t n k m")
a, b, c, d = symbols("a b c d")
f, g, h = symbols("f g h", cls=Function)
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def evaluate(expr_str):
    """Parse and evaluate a symbolic expression."""
    try:
        expr = parse_expr(expr_str, transformations=TRANSFORMS)
        simplified = simplify(expr)
        return {
            "input": expr_str,
            "parsed": str(expr),
            "simplified": str(simplified),
            "latex": latex(simplified),
            "numeric": str(N(simplified)) if simplified.is_number else None,
        }
    except Exception as e:
        return {"input": expr_str, "error": str(e)}


def differentiate(expr_str, var="x", order=1):
    """Compute derivative."""
    expr = parse_expr(expr_str, transformations=TRANSFORMS)
    v = Symbol(var)
    result = diff(expr, v, order)
    return {
        "input": expr_str,
        "variable": var,
        "order": order,
        "derivative": str(result),
        "simplified": str(simplify(result)),
        "latex": latex(result),
    }


def integrate_expr(expr_str, var="x", lower=None, upper=None):
    """Compute integral (definite or indefinite)."""
    expr = parse_expr(expr_str, transformations=TRANSFORMS)
    v = Symbol(var)
    if lower is not None and upper is not None:
        result = integrate(expr, (v, sympify(lower), sympify(upper)))
    else:
        result = integrate(expr, v)
    return {
        "input": expr_str,
        "variable": var,
        "bounds": [lower, upper] if lower is not None else "indefinite",
        "integral": str(result),
        "latex": latex(result),
    }


def solve_equation(expr_str, var="x"):
    """Solve equation (expr = 0)."""
    expr = parse_expr(expr_str, transformations=TRANSFORMS)
    v = Symbol(var)
    solutions = solve(expr, v)
    return {
        "input": f"{expr_str} = 0",
        "variable": var,
        "solutions": [str(s) for s in solutions],
        "latex_solutions": [latex(s) for s in solutions],
    }


def matrix_op(matrix_str, operation="eigenvals"):
    """Matrix operations — eigenvalues, determinant, inverse, etc."""
    mat = Matrix(json.loads(matrix_str))
    result = {"matrix": str(mat), "operation": operation}

    if operation == "eigenvals":
        evals = mat.eigenvals()
        result["eigenvalues"] = {str(k): v for k, v in evals.items()}
    elif operation == "eigenvects":
        evects = mat.eigenvects()
        result["eigenvectors"] = [
            {"eigenvalue": str(ev[0]), "multiplicity": ev[1], "vectors": [str(v) for v in ev[2]]}
            for ev in evects
        ]
    elif operation == "det":
        result["determinant"] = str(mat.det())
    elif operation == "inv":
        result["inverse"] = str(mat.inv())
    elif operation == "rank":
        result["rank"] = mat.rank()
    elif operation == "svd":
        U, S, V = mat.singular_value_decomposition()
        result["U"] = str(U)
        result["S"] = str(S)
        result["V"] = str(V)
    elif operation == "nullspace":
        ns = mat.nullspace()
        result["nullspace"] = [str(v) for v in ns]

    return result


def series_expand(expr_str, var="x", point=0, order=6):
    """Taylor/Laurent series expansion."""
    expr = parse_expr(expr_str, transformations=TRANSFORMS)
    v = Symbol(var)
    result = series(expr, v, point, order)
    return {
        "input": expr_str,
        "variable": var,
        "point": point,
        "order": order,
        "series": str(result),
        "latex": latex(result),
    }


def limit_expr(expr_str, var="x", point="oo"):
    """Compute limit."""
    expr = parse_expr(expr_str, transformations=TRANSFORMS)
    v = Symbol(var)
    pt = sympify(point)
    result = limit(expr, v, pt)
    return {
        "input": expr_str,
        "variable": var,
        "point": point,
        "limit": str(result),
        "latex": latex(result),
    }


def solve_ode(expr_str, func="f", var="x"):
    """Solve ordinary differential equation."""
    v = Symbol(var)
    fn = Function(func)(v)
    expr = parse_expr(expr_str, local_dict={func: Function(func), var: v},
                      transformations=TRANSFORMS)
    result = dsolve(expr, fn)
    return {
        "input": expr_str,
        "solution": str(result),
        "latex": latex(result),
    }


def information_theory(probs_str):
    """Compute entropy, KL divergence, mutual information from probability distributions."""
    probs = json.loads(probs_str)

    if isinstance(probs, list):
        # Shannon entropy
        p = [Rational(x).limit_denominator(1000) if isinstance(x, float) else Rational(x) for x in probs]
        H = -sum(pi * log(pi, 2) for pi in p if pi > 0)
        return {
            "distribution": probs,
            "entropy_bits": str(N(H)),
            "entropy_symbolic": str(H),
            "max_entropy": str(N(log(len(probs), 2))),
        }
    elif isinstance(probs, dict) and "p" in probs and "q" in probs:
        # KL divergence
        p = [Rational(x).limit_denominator(1000) if isinstance(x, float) else Rational(x) for x in probs["p"]]
        q = [Rational(x).limit_denominator(1000) if isinstance(x, float) else Rational(x) for x in probs["q"]]
        kl = sum(pi * log(pi / qi, 2) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
        return {
            "P": probs["p"],
            "Q": probs["q"],
            "kl_divergence_bits": str(N(kl)),
        }


def free_compute(code_str):
    """Execute arbitrary SymPy code and return the result."""
    local_ns = {
        "sympy": sympy, "symbols": symbols, "Symbol": Symbol, "Function": Function,
        "Matrix": Matrix, "Rational": Rational, "pi": pi, "E": E, "I": I, "oo": oo,
        "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "sqrt": sqrt,
        "diff": diff, "integrate": integrate, "solve": solve, "simplify": simplify,
        "limit": limit, "series": series, "summation": summation, "product": product,
        "factorial": factorial, "binomial": binomial, "gamma": gamma,
        "x": x, "y": y, "z": z, "t": t, "n": n, "k": k, "m": m,
        "a": a, "b": b, "c": c, "d": d,
        "latex": latex, "N": N, "Eq": Eq, "Ne": Ne,
        "print": print,
    }
    exec(code_str, {"__builtins__": {}}, local_ns)
    if "_result" in local_ns:
        return str(local_ns["_result"])
    return "Done (set _result = <expr> to return a value)"


def main():
    parser = argparse.ArgumentParser(description="Symbolic Math Engine")
    sub = parser.add_subparsers(dest="command")

    ev = sub.add_parser("eval", help="Evaluate/simplify expression")
    ev.add_argument("expr")

    df = sub.add_parser("diff", help="Differentiate")
    df.add_argument("expr")
    df.add_argument("--var", default="x")
    df.add_argument("--order", type=int, default=1)

    ig = sub.add_parser("integrate", help="Integrate")
    ig.add_argument("expr")
    ig.add_argument("--var", default="x")
    ig.add_argument("--lower", default=None)
    ig.add_argument("--upper", default=None)

    sl = sub.add_parser("solve", help="Solve equation (= 0)")
    sl.add_argument("expr")
    sl.add_argument("--var", default="x")

    mx = sub.add_parser("matrix", help="Matrix operations")
    mx.add_argument("matrix", help="JSON matrix, e.g. '[[1,2],[3,4]]'")
    mx.add_argument("--op", default="eigenvals", choices=["eigenvals", "eigenvects", "det", "inv", "rank", "svd", "nullspace"])

    se = sub.add_parser("series", help="Series expansion")
    se.add_argument("expr")
    se.add_argument("--var", default="x")
    se.add_argument("--point", type=int, default=0)
    se.add_argument("--order", type=int, default=6)

    lm = sub.add_parser("limit", help="Compute limit")
    lm.add_argument("expr")
    lm.add_argument("--var", default="x")
    lm.add_argument("--point", default="oo")

    it = sub.add_parser("entropy", help="Information theory (entropy, KL divergence)")
    it.add_argument("probs", help="JSON: [p1,p2,...] for entropy, {\"p\":[...],\"q\":[...]} for KL")

    fc = sub.add_parser("exec", help="Execute arbitrary SymPy code")
    fc.add_argument("code")

    args = parser.parse_args()

    if args.command == "eval":
        result = evaluate(args.expr)
    elif args.command == "diff":
        result = differentiate(args.expr, args.var, args.order)
    elif args.command == "integrate":
        result = integrate_expr(args.expr, args.var, args.lower, args.upper)
    elif args.command == "solve":
        result = solve_equation(args.expr, args.var)
    elif args.command == "matrix":
        result = matrix_op(args.matrix, args.op)
    elif args.command == "series":
        result = series_expand(args.expr, args.var, args.point, args.order)
    elif args.command == "limit":
        result = limit_expr(args.expr, args.var, args.point)
    elif args.command == "entropy":
        result = information_theory(args.probs)
    elif args.command == "exec":
        print(free_compute(args.code))
        return
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
