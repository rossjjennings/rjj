import numpy as np

class LatinSquare:
    def __init__(self, n):
        self.n = n
        self.proper = True
        self.improper_point = None
        self.extra_indices = None

        indices = np.arange(n)
        self.s_rc = (indices + indices[:, np.newaxis]) % n
        self.r_cs = (indices - indices[:, np.newaxis]) % n
        self.c_rs = (indices - indices[:, np.newaxis]) % n

    def validate(self):
        m, n = self.s_rc.shape
        assert m == n
        for r in range(n):
            for c in range(n):
                s = s_rc[r, c]
                assert r == r_cs[c, s]
                assert c == c_rs[r, s]

    def step(self, rng=np.default_rng()):
        if self.proper:
            r, c = rng.integers(n, size=2)
            s1 = self.s_rc[r, c]
            s = rng.choice(np.delete(np.arange(self.n), s1))
            r1 = self.r_cs[c, s]
            c1 = self.c_rs[r, s]
        else:
            r, c, s = self.improper_point
            rx, cx, sx = self.extra_indices
            r1 = rng.choice([self.r_cs[c, s], rx])
            c1 = rng.choice([self.c_rs[r, s], cx])
            s1 = rng.choice([self.s_rc[r, c], sx])
        self.s_rc[r, c] = s # was s1
        self.r_cs[c, s] = r
        self.c_rs[r, s] = c
        self.s_rc[r1, c] = s1 # was s
        self.r_cs[c, s1] = r1
        self.c_rs[r1, s1] = c
        self.s_rc[r, c1] = s1 # was s
        self.r_cs[c1, s1] = r
        self.c_rs[r, s1] = c1
        proper = (s1 == self.s_rc[r1, c1])
        if proper and self.proper:
            self.s_rc[r1, c1] = s # was s1
            self.r_cs[c1, s] = r1
            self.c_rs[r1, s] = c1
        elif self.proper:
            self.improper_point = (r1, c1, s1)
            self.extra_indices = (r, c, s)
