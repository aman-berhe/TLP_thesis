Computational Learning Theory

PAC Learning Leslie Valiat

c=[a,b]
def guessInterval():
    PAC=probability-approximatelt+correct
    err=Px(h(x)!=c(x))>1-epsilon
    if epsilon big:
        print("allowed to make more errors")
    else:
        print("make it as exact as possible")

"""
3 term DNF fomulas can not belearned using PAC models
proof 3-coloring of grph
3-CNF are PAC-learnable
we can transform 3-DNF into 3-CNF
"""

"""
Sample Coplexity :
hwo many examples are need to find PAC solution
--> depeneds on the error (epsilon properties of H)
Chernoff Bound
The larger the complexity the more examples you need
"""
