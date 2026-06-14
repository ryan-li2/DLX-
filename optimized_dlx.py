class Node:
    __slots__ = (
        "L",
        "R",
        "U",
        "D",
        "C",
        "RH",
        "row_id",
        "name",
        "color",
        "row_header",
        "col_header",
    )

    def __init__(self):
        self.L = self.R = self.U = self.D = self
        self.C = None
        self.RH = None
        self.row_id = None
        self.name = None
        self.color = None
        self.row_header = False
        self.col_header = False


class Column(Node):
    __slots__ = ()

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.col_header = True


class Row(Node):
    __slots__ = ()

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.row_header = True


class OptimizedDLX:
    """
    Drop-in DLX+ implementation with lower object overhead and correct undo.

    The search remains the same bounded heuristic as the original algorithm;
    it is not an exact solver unless all alternatives are explored.
    """

    __slots__ = (
        "root",
        "columns",
        "rows",
        "D",
        "LB",
        "N_max",
        "backtrack_nodes",
        "solution_size",
        "solution",
        "best_solution",
        "current",
    )

    def __init__(self, n_cols, rows, row_idx, D, LB, N_max, backtrack_node):
        root = Column("root")
        root.name = float("inf")
        self.root = root
        self.columns = [Column(i) for i in range(n_cols)]
        self.rows = [Row(i) for i in range(len(rows))]
        self.D = D
        self.LB = LB
        self.N_max = N_max
        self.backtrack_nodes = backtrack_node
        self.solution_size = 0
        self.solution = []
        self.best_solution = []
        self.current = 0

        last = root
        for column in self.columns:
            column.L, column.R = last, root
            last.R = column
            root.L = column
            last = column

        last_row = root
        columns = self.columns
        for r_idx, row_columns in enumerate(rows):
            row_header = self.rows[r_idx]
            row_header.U, row_header.D = last_row, root
            last_row.D = row_header
            root.U = row_header
            last_row = row_header
            row_header.C = row_header
            row_header.color = row_idx[r_idx][0]

            first = None
            for column_idx in row_columns:
                column = columns[column_idx]
                node = Node()
                node.C = column
                node.row_id = r_idx
                node.RH = row_header

                node.U, node.D = column.U, column
                column.U.D = node
                column.U = node

                if first is None:
                    first = node
                    node.L = node.R = node
                else:
                    node.L, node.R = first.L, first
                    first.L.R = node
                    first.L = node

            if first is None:
                row_header.R = row_header.L = row_header
            else:
                row_header.R = first
                row_header.L = first.L
                first.L.R = row_header
                first.L = row_header

    @staticmethod
    def uncover(column):
        i = column.U
        while i is not column:
            j = i.L
            while j is not i:
                j.D.U = j
                j.U.D = j
                j = j.L
            i = i.U

    @staticmethod
    def cover(column):
        i = column.D
        while i is not column:
            j = i.R
            while j is not i:
                j.D.U = j.U
                j.U.D = j.D
                j = j.R
            i = i.D

    def check_UB(self, candidate):
        demand = self.D
        upper_bound = demand[candidate.row_id]
        color = candidate.RH.color
        row = candidate.RH.D
        remaining = self.N_max - self.solution_size
        count = 1
        root = self.root

        while row is not root and count < remaining:
            row_color = row.color
            if row_color > color:
                upper_bound += demand[row.name]
                color = row_color
                count += 1
            row = row.D

        return upper_bound

    def row_loop(self, candidate):
        self.cover(candidate.C)
        row_id = candidate.RH.name
        self.solution.append(row_id)
        self.solution_size += 1
        self.current += self.D[row_id]

        node = candidate.R
        while node is not candidate:
            if not node.row_header:
                self.cover(node.C)
            node = node.R

    def uncover_row_cols(self, candidate):
        node = candidate.L
        while node is not candidate:
            if not node.row_header:
                self.uncover(node.C)
            node = node.L

        row_id = self.solution.pop()
        self.current -= self.D[row_id]
        self.solution_size -= 1

    def _record_incumbent(self):
        if self.current > self.LB:
            self.LB = self.current
            self.best_solution = self.solution.copy()

    def _search(self):
        root = self.root
        row = root.D

        if row is root or self.solution_size >= self.N_max:
            self._record_incumbent()
            return

        last_name = self.solution[-1]
        while row is not root and row.name < last_name:
            row = row.D
        if row is root:
            self._record_incumbent()
            return

        candidate = row.R
        if self.current + self.check_UB(candidate) <= self.LB:
            return

        self.row_loop(candidate)
        self._search()
        self.uncover_row_cols(candidate)
        self.uncover(candidate.C)

    def _backtrack_next_node(self, candidate):
        color = candidate.RH.color
        alternative = candidate.D

        while not alternative.col_header:
            if (
                alternative.RH.color != color
                and self.current + self.check_UB(alternative) > self.LB
            ):
                self.row_loop(alternative)
                self._search()
                self.uncover_row_cols(alternative)
                self.uncover(alternative.C)
                break
            alternative = alternative.D

        self.uncover(candidate.C)

    def _search_range(self):
        row = self.root.D
        if (
            row is self.root
            or self.solution_size >= self.backtrack_nodes
            or self.solution_size >= self.N_max
        ):
            self._record_incumbent()
            return

        candidate = row.R
        self.row_loop(candidate)
        self._search_range()
        self.uncover_row_cols(candidate)
        self._backtrack_next_node(candidate)

    def search_range(self):
        """Run a depth setting and yield at most one improved incumbent."""
        previous_bound = self.LB
        self._search_range()
        if self.LB > previous_bound:
            yield self.best_solution, self.LB


# Drop-in name used by the original experiment code.
DLX = OptimizedDLX
