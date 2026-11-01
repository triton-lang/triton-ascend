"""Direct aliases for a Triton wheel built with the accompanying PR.

Use frontend='compat' on the measured 8ca956c1 wheel. There is no automatic
fallback: choosing 'native' with an unpatched wheel fails visibly.
"""
from triton.language.extra.tla.cube import mmad as matmul
from triton.language.extra.tla.simd import vexpdif, vmuls
from triton.language.extra.tla.view import prefix_view
