import numpy as np

def extract_formula(layer, threshold=0.01):
    in_dim = layer.in_features
    out_dim = layer.out_features
    degree = layer.degree

    print("\n" + "=" * 50)
    print("EXTRACTED SYMBOLIC FORMULAS FROM KUMONOSU")
    print("=" * 50)

    for o in range(out_dim):
        terms = []

        bias_val = layer.b[0, o]
        if abs(bias_val) > threshold:
            terms.append(f"{bias_val:+.3f}")

        for i in range(in_dim):
            for d in range(degree + 1):
                coeff = layer.C[i, o, d]

                if abs(coeff) > threshold:
                    if d == 0:
                        terms.append(f"{coeff:+.3f}")
                    elif d == 1:
                        terms.append(f"{coeff:+.3f}*(x_{i})")
                    else:
                        terms.append(f"{coeff:+.3f}*(x_{i}^{d})")

        formula_str = " ".join(terms) if terms else "0.0"
        if formula_str.startswith("+"):
            formula_str = formula_str[1:].strip()

        print(f"\nNode_{o}(X) = {formula_str}")

    print("=" * 50 + "\n")