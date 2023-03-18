import numpy as np

class LatinSquare:
    def __init__(self, square):
        """
        Construct row-symbol and column-symbol views for a Latin square, verifying the
        Latin square property in the process.
        """
        if not len(square.shape) == 2 and square.shape[0] == square.shape[1]:
            raise ValueError("Input array was not square")

        self.improper_point = None
        self.size = square.shape[0]
        self.row_col = np.array(square, dtype=object)

        row_sym = []
        for row in square:
            col_by_sym = []
            for s in range(self.size):
                idx, = np.where(row == s)
                if len(idx) == 1:
                    col_by_sym.append(idx[0])
                else:
                    raise ValueError("Input array was not a Latin square")
            row_sym.append(col_by_sym)
        self.row_sym = np.array(row_sym, dtype=object)

        col_sym = []
        for col in square.T:
            row_by_sym = []
            for s in range(self.size):
                idx, = np.where(col == s)
                if len(idx) == 1:
                    row_by_sym.append(idx[0])
                else:
                    raise ValueError("Input array was not a Latin square")
            col_sym.append(row_by_sym)
        self.col_sym = np.array(col_sym, dtype=object)

    def take_step(self, row, col, sym):
        """
        Make a "±1-move", after Jacobsen & Matthews (1996), J. Comb. Des. 4 (6), p. 410.
        If the incidence cube is initially proper, (row, col, sym) defines the initial 0-cell,
        and we should have ~(row, col, sym). If the cube is initially improper, (row, col, sym)
        defines the choice of proper point along each axis. In this case, there are two
        possible values for each of row, col, and sym, corresponding to the duplicate points
        along the same line as the improper point.
        """

        # Define sub-cube
        if self.improper_point is None:
            # cube is initially proper
            if self.row_col[row, col] == sym:
                raise ValueError("Not a valid move")
            alt_sym = self.row_col[row, col]
            alt_col = self.row_sym[row, sym]
            alt_row = self.col_sym[col, sym]

            self.row_col[row, col] = sym
            self.row_sym[row, sym] = col
            self.col_sym[col, sym] = row
        else:
            # cube is initially improper
            alt_row, alt_col, alt_sym = row, col, sym
            row, col, sym = self.improper_point

            s1, s2 = self.row_col[row, col].proper_values
            if alt_sym == s1:
                self.row_col[row, col] = s2
            elif alt_sym == s2:
                self.row_col[row, col] = s1
            else:
                raise ValueError("Not a valid move")

            c1, c2 = self.row_sym[row, sym].proper_values
            if alt_col == c1:
                self.row_sym[row, sym] = c2
            elif alt_col == c2:
                self.row_sym[row, sym] = c1
            else:
                raise ValueError("Not a valid move")

            r1, r2 = self.col_sym[col, sym].proper_values
            if alt_row == r1:
                self.col_sym[col, sym] = r2
            elif alt_row == r2:
                self.col_sym[col, sym] = r1
            else:
                raise ValueError("Not a valid move")

        # "Proper" updates
        self.row_col[row, alt_col] = alt_sym
        self.row_col[alt_row, col] = alt_sym

        self.row_sym[row, alt_sym] = alt_col
        self.row_sym[alt_row, sym] = alt_col

        self.col_sym[col, alt_sym] = alt_row
        self.col_sym[alt_col, sym] = alt_row

        # "Improper" update
        if self.row_col[alt_row, alt_col] == alt_sym:
            # cube becomes proper
            self.improper_point = None
            self.row_col[alt_row, alt_col] = sym
            self.row_sym[alt_row, alt_sym] = col
            self.col_sym[alt_col, alt_sym] = row
        else:
            # cube becomes improper
            self.improper_point = (alt_row, alt_col, alt_sym)
            old_sym = self.row_col[alt_row, alt_col]
            old_col = self.row_sym[alt_row, alt_sym]
            old_row = self.col_sym[alt_col, alt_sym]
            self.row_col[alt_row, alt_col] = ImproperEntry((old_sym, sym), alt_sym)
            self.row_sym[alt_row, alt_sym] = ImproperEntry((old_col, col), alt_col)
            self.col_sym[alt_col, alt_sym] = ImproperEntry((old_row, row), alt_row)

    def random_step(self, rng=np.random.default_rng()):
        """
        Take a random, valid step (±1-move) using take_step().
        """
        if self.improper_point is None:
            row, col = rng.integers(self.size, size=2)
            cur_sym = self.row_col[row, col]
            sym = rng.choice(list(set(range(self.size)) - {cur_sym}))
        else:
            r, c, s = self.improper_point
            sym = rng.choice(self.row_col[r, c].proper_values)
            col = rng.choice(self.row_sym[r, s].proper_values)
            row = rng.choice(self.col_sym[c, s].proper_values)
        self.take_step(row, col, sym)

    def step_until_proper(self, rng=np.random.default_rng()):
        """
        Take random steps until the square is proper.
        """
        while True:
            self.random_step(rng)
            if self.improper_point is None:
                break

    @classmethod
    def cyclic(cls, size):
        square = (np.arange(size)[:, np.newaxis] + np.arange(size)) % size
        return cls(square)

    @classmethod
    def random(cls, size, rng=np.random.default_rng()):
        ls = cls.cyclic(size)
        for i in range(size**3):
            ls.random_step(rng)
        ls.step_until_proper()
        return ls

class ImproperEntry:
    def __init__(self, proper_values, improper_value):
        self.proper_values = proper_values
        self.improper_value = improper_value

    def __repr__(self):
        return "{0}+{1}-{2}".format(*self.proper_values, self.improper_value)
