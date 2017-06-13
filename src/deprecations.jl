#
# deprecations.jl --
#
# Deprecations in OptimPack module.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2017, Éric Thiébaut.
#

import Base: @deprecate

# Deprecated in v0.4.0
@deprecate fmin_global OptimPack.Bradi.minimize
