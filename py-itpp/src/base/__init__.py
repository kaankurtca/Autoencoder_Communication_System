# -------------------------------------------------------------------------
#
# Copyright (C) 2016 CC0 1.0 Universal (CC0 1.0)
#
# The person who associated a work with this deed has dedicated the work to
# the public domain by waiving all of his or her rights to the work
# worldwide under copyright law, including all related and neighboring
# rights, to the extent allowed by law.
#
# You can copy, modify, distribute and perform the work, even for commercial
# purposes, all without asking permission.
#
# See the complete legal text at
# <https://creativecommons.org/publicdomain/zero/1.0/legalcode>
#
# -------------------------------------------------------------------------

from . import math

from .binary import *
from .mat import *
from .matfunc import *
from .random import *
from .specmat import *
from .vec import *

from .mat import elem_mult as elem_mult_mat
from .vec import elem_mult as elem_mult_vec

from .mat import elem_div as elem_div_mat
from .vec import elem_div as elem_div_vec
