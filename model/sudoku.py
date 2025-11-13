import cpmpy as cp

class SudokuProblem():

    def __init__(self, array):
        self.array = array

    def make_model(self):
        constraints = []
        fact_to_explain = []
        dict_constraint_clues = {}

        nrow = ncol = len(self.array)
        n = int(nrow ** (1 / 2))
        cells = cp.intvar(1, nrow, shape=(nrow, ncol), name="cells")

        for i, row in enumerate(cells):
            row_con = cp.AllDifferent(row)

            constraints.append(row_con)
            dict_constraint_clues[str(row_con)] = f'ROW_{i}'


        for i, col in enumerate(cells.T):
            col_con = cp.AllDifferent(col)

            constraints.append(col_con)
            dict_constraint_clues[str(col_con)] = f'COL_{i}'

        count=1
        for i in range(0, nrow, n):
            for j in range(0, ncol, n):
                block_con = cp.AllDifferent(cells[i:i + n, j:j + n])

                constraints.append(block_con)
                dict_constraint_clues[str(block_con)] = f'BLOCK_{count}'
                count+=1

        for i in range(nrow):
            for j in range(ncol):
                if self.array[i][j] == 'e':
                    fact_to_explain.append(cells[i,j])
                else:
                    constraints.append(cells[i,j] == self.array[i][j])


        return [], constraints, fact_to_explain, dict_constraint_clues