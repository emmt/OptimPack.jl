# Version 0.4.0

- This version provides implementation of vectorized operations, line search
  and quasi-Newton methods written in pure Julia.

- The wrapper to the C library has moved into sub-module `OptimPack.CLib`.

- Provides an implementation of the STEP (Select The Easiest Point) method for
  global univariate optimization:

> Swarzberg, S., Seront, G. & Bersini, H., "*S.T.E.P.: the easiest way to
> optimize a function,*" in IEEE World Congress on Computational Intelligence,
> Proceedings of the First IEEE Conference on Evolutionary Computation,
> vol. **1**, pp. 519-524 (1994).


# Version 0.3.0

- add compatibility with C library OptimPack 3.0;
