import copy
import os
import random
import numpy as np
import time
import operator
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# region
BOARD_SIZE = 5
DEPTH = 2
COMBINATION_RESTRICTION = 10  # Twelve is the maximum for vicareum
# ___________________GREADY HEURISTIC__________________________________________________
INITVALUE = 0.5
ITWOLIBERTIES = 0.48
ITHREELIBERTIES = 0.49

BORDERVALUE = 0.035
QUADRANTVALUE = 0.05

HASNEIGBOR = 0.095

GROUPVALUE = 0.2
RINGVALUE = 0.215
CIRCLEVALUE = 0.145
DIAMONDVALUE = 0.2
HALFDIAMONDVALUE = 0.115
SIDEBREAKER = 0.055
SINGLEENEMYVALUE = 0.02
CLOSEPIPEVALUE = 0.225
CLOSEUPVALUE = 0.12
GRIDVALUE = 0.23

WIPE_MAGNITUDE = 0.6
SAVEMAGNITUDE = 0.45
CIRCLEMAGNITUDE = 0.055
SCOREBASE = 0.55

CIRCLEDEPTHDISCARD = 0.5
FUTUREMOVE_DISCARD = 0.6

AMONGALLIESPENALTY = -0.2  # 95% when -0.2
AMONGENEMYPENALTY = -0.15  # TODO: TURN BACK TO 0.2
PROBABLEDEATHPENALTY = -0.5
PERILOUSPOSITIONDISCARD = 0.4  # TODO: turn back to 0.1
OPPONENTSSAVEEVALUATIONVALUE = 0.1
TAKINGOVERAROWSCORE = 0.12

POTENTIALTARGETSCORE = 0.3

LIBERTYVALUE = 0.085
CLOSETOPREVMOVEVALUE = 0.185

WINNER_SCORE = 100
LOSER_SCORE = -100
PASSSCORE = -20
# ___________________QLEARNER_____________________________________________________________

IVAL = 0.5

ANYMOVE = (101, 101)


# endregion
class Game:
    def __init__(self, N):
        self.N = N
        self.prev_move_file = 'prev_move.txt'
        self.current_state = np.zeros((N, N), dtype=np.int)
        self.prev_state = np.zeros((N, N), dtype=np.int)
        self.player_label = None
        self.opponent_label = None
        self.prev_move = None
        self.prev_opponents_move = None
        self.n_move = 0
        self.KO = None
        self.wiped_chips = []
        self.max_move = (N * N - 1) - 1  # coz count from 0
        self.game_start = [0, 1, 2, 3, 4, 5]
        self.depth_limit = 5

    def read_states(self, state_file='input.txt'):
        with open(state_file, 'r') as states:
            self.player_label = int(states.readline())
            if self.player_label == 1:
                self.opponent_label = 2
            else:
                self.opponent_label = 1

            two_states = states.readlines()
            ps = two_states[:self.N]
            cs = two_states[self.N:]

            for i in range(self.N):
                self.current_state[i] = np.array(list(cs[i].strip()))
                self.prev_state[i] = np.array(list(ps[i].strip()))

        self.init_prev_move()
        self.init_prev_opponents_move()
        self.init_dead_chips()
        self.assign_depth_limit()

    def set_depth(self):
        if self.n_move in self.game_start:
            return DEPTH
        else:
            possible_combinations = self.number_of_empty_cells()
            if possible_combinations >= COMBINATION_RESTRICTION:
                return DEPTH
            else:
                restriction_combinations = (self.N * self.N) * (self.N * self.N - 1)
                found_limit = DEPTH
                while possible_combinations * (possible_combinations - 1) < restriction_combinations:
                    if found_limit >= self.depth_limit:
                        return self.depth_limit
                    found_limit += 1
                    possible_combinations *= (possible_combinations - 1)
                return found_limit

    def number_of_empty_cells(self):
        return np.count_nonzero(self.current_state == 0)

    def move_number_incrementation(self):
        self.n_move += 1

    def is_the_move_last(self):
        if self.n_move >= self.max_move:
            return True
        return False

    def swap_players(self):
        self.player_label, self.opponent_label = self.opponent_label, self.player_label

    def is_end_of_game(self, is_action_pass=False):
        if self.is_the_move_last():
            return True
        if self.are_states_identical(self.current_state, self.prev_state) and is_action_pass:
            return True
        return False

    def init_prev_move(self):
        try:
            with open(self.prev_move_file, 'r') as info:
                line = info.readline()
                last_player = int(line[0])
                if self.player_label == last_player:
                    self.init_from_file_info(line)
                else:
                    self.init_new_game_info()
        except FileNotFoundError:
            self.init_new_game_info()

    def init_prev_opponents_move(self):
        state_diff = self.current_state - self.prev_state
        if np.count_nonzero(state_diff == 0) == 25:
            return
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if state_diff[i][j] == self.opponent_label:
                        self.prev_opponents_move = (i, j)

    def get_prev_opponents_move(self):
        state_diff = self.current_state - self.prev_state
        if np.count_nonzero(state_diff == 0) == 25:
            return
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if state_diff[i][j] == self.opponent_label:
                        return (i, j)

    def are_states_identical(self, state1, state2):
        for i in range(self.N):
            for j in range(self.N):
                if state1[i][j] != state2[i][j]:
                    return False
        return True

    def _inner_cells(self):
        return {
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
            (3, 1), (3, 2), (3, 3)}

    def _outer_cells(self):
        return {
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 4),
            (2, 0), (2, 4),
            (3, 0), (3, 4),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        }

    def init_from_file_info(self, info_line):
        tokens = str.split(info_line, " ")
        self.n_move = int(str.strip(tokens[1]))  # ???
        if tokens[2] == 'PASS':
            self.prev_move = 'PASS'
        else:
            prev_row, prev_col = str.split(tokens[2], ",")
            self.prev_move = (int(prev_row), int(prev_col))

    def init_new_game_info(self):
        if self.player_label == 1:
            self.n_move = 0
        else:
            self.n_move = 1

    def commit_move_info(self, move_row, move_col):
        with open(self.prev_move_file, 'w') as info:
            info.write(str(self.player_label) + " "
                       + str((self.n_move + 2)) + " "
                       + str(move_row) + "," + str(move_col))

    def assign_depth_limit(self):
        if (self.max_move - self.n_move) > 6:
            self.depth_limit = 6
        else:
            self.depth_limit = self.max_move - self.n_move

    def current_copy(self):
        return copy.deepcopy(self.current_state)

    def prev_copy(self):
        return copy.deepcopy(self.prev_state)

    def get_adjacent_allies_and_empty_cells(self, i, j, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        neighbors = self.get_all_neighbors(i, j)
        group = self.find_group_with_empty_cells(neighbors, player_label, altboard)
        return group

    def find_group_with_empty_cells(self, chips, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        group = []

        for chip in chips:
            if board[chip[0]][chip[1]] == player_label or board[chip[0]][chip[1]] == 0:
                group.append(chip)
        return group

    def find_group(self, chips, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        group = []

        for chip in chips:
            if board[chip[0]][chip[1]] == player_label:
                group.append(chip)
        return group

    def is_cell_on_border(self, cell):
        if cell in self._outer_cells():
            return True
        return False

    def get_all_neighbors(self, i, j):
        neighbors = []

        if i > 0: neighbors.append((i - 1, j))
        if i < self.N - 1: neighbors.append((i + 1, j))

        if j > 0: neighbors.append((i, j - 1))
        if j < self.N - 1: neighbors.append((i, j + 1))
        return neighbors

    def get_diagonal_neigbors(self, i, j):
        neigbors = []

        if i > 0 and j < self.N - 1: neigbors.append((i - 1, j + 1))
        if i > 0 and j > 0: neigbors.append((i - 1, j - 1))
        if i < self.N - 1 and j > 0: neigbors.append((i + 1, j - 1))
        if i < self.N - 1 and j < self.N - 1: neigbors.append((i + 1, j + 1))
        return neigbors

    def get_adjacent_allies(self, i, j, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        neighbors = self.get_all_neighbors(i, j)
        group = self.find_group(neighbors, player_label, altboard)
        return group

    def get_adjacent_diagonal_allies(self, i, j, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        dgn = self.get_diagonal_neigbors(i, j)
        group = self.find_group(dgn, player_label, altboard)
        return group

    def find_allies(self, i, j, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        frontier = [(i, j)]
        adjacent_chips = []

        while frontier:
            chip = frontier.pop()
            adjacent_chips.append(chip)
            adjacent_allies = self.get_adjacent_allies(chip[0], chip[1], player_label, altboard)

            for ally in adjacent_allies:
                if ally not in frontier and ally not in adjacent_chips:
                    frontier.append(ally)

        return adjacent_chips

    def valid_liberty(self, i, j, player_label, altboard=[]):
        if len(altboard) == 0:
            board = self.current_copy()
        else:
            board = altboard

        adjacent_allies = self.find_allies(i, j, player_label, altboard)

        for ally in adjacent_allies:
            neighbors = self.get_all_neighbors(ally[0], ally[1])
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                    return True
        return False

    def init_dead_chips(self):
        diff_board = self.current_state - self.prev_state
        for i in range(self.N):
            for j in range(self.N):
                if diff_board[i][j] == -self.player_label:
                    self.wiped_chips.append((i, j))

    def liberty_second_phase(self, i, j, opponent_label, player_label=False):
        board = self.current_copy()
        if player_label == False:
            board[i][j] = self.player_label
        else:
            board[i][j] = player_label

        chips_to_wipe = self.chips_to_wipe(opponent_label, board)
        if len(chips_to_wipe) == 0:
            return board
        self.wipe(chips_to_wipe, board)

        return board

    def liberty_second_phase_with_score(self, i, j, opponent_label):
        board = self.current_copy()
        board[i][j] = self.player_label
        chips_to_wipe = self.chips_to_wipe(opponent_label, board)
        if len(chips_to_wipe) == 0:
            return board, 0
        self.wipe(chips_to_wipe, board)
        return board, chips_to_wipe

    def wipe(self, positions, board):
        for chip in positions:
            board[chip[0]][chip[1]] = 0

    def chips_to_wipe(self, label, board):
        wipe = []

        for i in range(self.N):
            for j in range(self.N):
                if board[i][j] == label:
                    if not self.valid_liberty(i, j, label, board):
                        wipe.append((i, j))
        return wipe

    def is_move_valid(self, i, j, opponent_check=False):
        board = self.current_state

        player_label = self.player_label
        opponent_label = self.opponent_label

        if opponent_check:
            player_label = self.opponent_label
            opponent_label = self.player_label

        # check if the cell belongs to boards space
        if not ((0 <= i < self.N) and (0 <= j < self.N)):
            return False

        if board[i][j] != 0:  #
            return False

        test_board = self.current_copy()
        test_board[i][j] = player_label
        if self.valid_liberty(i, j, player_label, test_board):
            return True
        if opponent_check:
            board = self.liberty_second_phase(i, j, opponent_label, player_label=player_label)
            if not self.valid_liberty(i, j, player_label, board):
                return False
        else:
            board = self.liberty_second_phase(i, j, opponent_label)
            if not self.valid_liberty(i, j, player_label, board):
                return False

        if i == 0 and j == 3:
            asdasd = 1

        if self.wiped_chips and self.are_states_identical(self.prev_state, board):
            self.KO = self.wiped_chips
            return False

        return True

    def get_available_moves(self, shrink=False):
        available_moves = []
        if shrink == True:
            for row, col in self.shrink_the_board():
                if self.is_move_valid(row, col):
                    available_moves.append((row, col))
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if self.is_move_valid(i, j):
                        available_moves.append((i, j))
        return available_moves

    def shrink_the_board(self):
        if self.number_of_empty_cells() == self.N * self.N:
            return self._inner_cells()
        else:
            cells = set(self._inner_cells())
            for outer_row, outer_col in self._outer_cells():
                if self.current_state[outer_row][outer_col] != 0:
                    nbs = self.get_all_neighbors(outer_row, outer_col)
                    dgs = self.get_diagonal_neigbors(outer_row, outer_col)
                    cells.add((outer_row, outer_col))
                    for cell in nbs:
                        cells.add(cell)
                    for cell in dgs:
                        cells.add(cell)
            return cells

    def distance_to_move(self, row, col, prev_move=True):
        if prev_move:
            if self.prev_move is None or self.prev_move == "PASS":
                return np.inf
            else:
                prow, pcol = self.prev_move
                return abs((prow - row)) + abs((pcol - col))

    def apply_move(self, row, col):
        new_board = self.current_copy()
        new_board[row][col] = self.player_label
        wiped = self.chips_to_wipe(self.opponent_label, new_board)
        if len(wiped) > 0:
            self.wipe(wiped, new_board)
        self.current_state = new_board

    def init_from_state_string(self, string_state):
        self.player_label = int(string_state[0])

        if self.player_label == 1:
            self.opponent_label = 2
        else:
            self.opponent_label = 1

        string_state = string_state[1:]
        result = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                result[i][j] = string_state[BOARD_SIZE * i + j]
        self.current_state = result

    def is_potential_score(self, i, j):
        second_board, wiped_chips = self.liberty_second_phase_with_score(i, j, self.opponent_label)
        if self.valid_liberty(i, j, self.player_label, second_board):
            return wiped_chips
        return False

    def is_potential_loss(self, i, j):
        board = self.current_copy()
        board[i][j] = self.opponent_label
        chips_to_wipe = self.chips_to_wipe(self.player_label, board)
        if len(chips_to_wipe) > 0:
            return len(chips_to_wipe)
        return False

    def get_winner_label(self):
        white_chips = self.N / 2
        black_chips = 0

        for row in range(self.N):
            for col in range(self.N):
                if self.current_state[row][col] == 1:
                    black_chips += 1
                elif self.current_state[row][col] == 2:
                    white_chips += 1

        if white_chips > black_chips:
            return 2
        return 1

    def get_score_difference(self):
        wpoints = self.N / 2
        bpoints = 0

        for row in range(self.N):
            for col in range(self.N):
                if self.current_state[row][col] == 1:
                    bpoints += 1
                elif self.current_state[row][col] == 2:
                    wpoints += 1
        return abs(wpoints - bpoints)


class NegMaxAgent:
    def __init__(self, depth, root_game):
        self.max_depth = depth
        self.root_environment = root_game
        self.forbidden_move = None
        self.history = {}
        self.KO = None
        self.circles = {}
        for layer in range(depth):
            self.history[layer] = {}

    def opposite_player(self, player):
        if player == 1:
            return 2
        return 1

    def get_move(self):
        state_string = self.state_to_string(self.root_environment.current_state,
                                            self.root_environment.player_label)

        expected_score, move = self.negmax(0,
                                           state_string,
                                           self.root_environment.prev_copy(),
                                           self.root_environment.n_move,
                                           True,
                                           (-1, -1),
                                           self.root_environment.prev_move)

        return expected_score, move

    def negmax(self, current_depth, state, prev_state, move_number, is_my_turn, parent_move, my_previous_move):
        current_env = Game(self.root_environment.N)  #
        current_env.init_from_state_string(state)  #
        current_env.prev_state = prev_state  #
        current_env.n_move = move_number  # TODO: вынести всю эту бандуру в отдельную функцию и ниже тож
        current_env.init_prev_opponents_move()  #
        current_env.prev_move = my_previous_move  #
        current_env.init_dead_chips()  #
        moveset = current_env.get_available_moves()  #

        if current_depth not in self.history:
            self.history[current_depth] = {}

        if parent_move not in self.history[current_depth]:
            self.history[current_depth][parent_move] = {}

        if current_env.is_end_of_game():
            return self.get_best_evaluated_move(current_env, moveset, current_depth, parent_move, True)
        elif current_depth == self.max_depth:
            return self.get_best_evaluated_move(current_env, moveset, current_depth, parent_move)
        elif moveset == []:
            self.history[current_depth][parent_move][ANYMOVE] = PASSSCORE

        expected_value = -np.inf
        suggested_move = None
        if is_my_turn:
            expected_value = -np.inf

        for mrow, mcol in moveset:
            next_state_string = self.create_new_state_string(mrow, mcol, current_env)  # make move here
            next_env = Game(self.root_environment.N)
            next_env.init_from_state_string(next_state_string)
            next_env.n_move = current_env.n_move
            next_env.move_number_incrementation()  # move+=1
            next_env.prev_state = current_env.current_copy()  # !!!!!!!!!!!!
            next_env.init_prev_opponents_move()
            next_env.prev_move = current_env.prev_opponents_move  # !!!!! Here is where the prev move from him is passed
            next_env.init_dead_chips()

            self.get_best_evaluated_move(current_env, [(mrow, mcol)], current_depth, parent_move)

            next_move_score, next_move = self.negmax(current_depth + 1,
                                                     next_state_string,
                                                     current_env.current_copy(),
                                                     next_env.n_move,
                                                     not is_my_turn,
                                                     (mrow, mcol),
                                                     current_env.prev_opponents_move
                                                     )

        return expected_value, suggested_move

    def create_new_state_string(self, row, col, current_environment):
        new_state = current_environment.current_copy()
        new_state[row][col] = current_environment.player_label
        chips_to_wipe = current_environment.chips_to_wipe(current_environment.opponent_label, new_state)
        if len(chips_to_wipe) > 0:
            new_state = self.wipe(chips_to_wipe, new_state)

        return self.state_to_string(new_state, current_environment.opponent_label)

    def state_to_string(self, state, player):
        return str(player) + ''.join([str(state[i][j]) for i in range(self.root_environment.N)
                                      for j in range(self.root_environment.N)]
                                     )

    def wipe(self, chips_to_wipe, board):
        if chips_to_wipe:
            for row, col in chips_to_wipe:
                board[row][col] = 0

        return board

    def is_death_probable(self, row, col, env, nbs):
        max_liberty = len(nbs)
        number_of_enemies = 0
        number_of_zeroes = 0
        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.opponent_label:
                number_of_enemies += 1
            elif env.current_state[nrow][ncol] == 0:
                number_of_zeroes += 1

        if number_of_zeroes == 1 and number_of_enemies == (max_liberty - 1):
            return True

    def get_group_score(self, row, col, env):
        altboard = env.current_copy()
        initial_state = env.current_copy()
        altboard[row][col] = env.player_label
        neigbors = env.get_all_neighbors(row, col)
        neigbors_to_check = []
        neigbor_group_liberties = []
        existing_groups = set()
        score = 0
        for neigbor in neigbors:
            nrow, ncol = neigbor
            if altboard[nrow][ncol] == env.opponent_label:
                neigbors_to_check.append(neigbor)

        for neigbor in neigbors_to_check:
            neigbor_group_liberties, group_size = self.number_of_enemy_group_liberties_with_group_size(neigbor[0],
                                                                                                       neigbor[1], env,
                                                                                                       altboard)
            is_new_group = False
            if len(group_size) > 0:
                if not group_size.issubset(existing_groups):
                    existing_groups.add(tuple(group_size))
                    is_new_group = True
            if len(group_size) == 1:
                if self.is_rekillable(row, col, env, group_size.pop()):
                    continue
            if len(group_size) == 1 and list(group_size)[0] in [(0, 0), (0, 4), (4, 0), (4, 4)]:
                continue
            elif len(neigbor_group_liberties) <= 2:
                if is_new_group:
                    score += len(group_size) * (WIPE_MAGNITUDE / 4)
        if len(existing_groups) > 0:
            if len(neigbor_group_liberties) == 0:
                return WIPE_MAGNITUDE * len(group_size)
            else:
                return self.being_on_border_non_linearity(row, col) * self.get_group_non_linearity(
                    len(neigbor_group_liberties))
        else:
            return 0

    def save_group_score(self, row, col, env):
        initial_state = env.current_copy()
        altboard = env.current_copy()
        altboard[row][col] = env.opponent_label
        neigbors = env.get_all_neighbors(row, col)
        neigbors_to_check = []
        existing_groups = set()
        score = 0
        for neigbor in neigbors:
            nrow, ncol = neigbor
            if altboard[nrow][ncol] == env.player_label:
                neigbors_to_check.append(neigbor)

        for neigbor in neigbors_to_check:
            is_new_group = False
            neigbor_group_liberties, group_size = self.number_of_group_liberties_with_froup_size(neigbor[0], neigbor[1],
                                                                                                 env, altboard)
            if len(group_size) > 0:
                if not group_size.issubset(existing_groups):
                    existing_groups.add(tuple(group_size))
                    is_new_group = True
            if len(group_size) == 1 and list(group_size)[0] in [(0, 0), (0, 4), (4, 0), (4, 4)]:
                continue
            elif len(neigbor_group_liberties) <= 1:
                if is_new_group:
                    initial_group_liberties, initial_group_size = self.number_of_group_liberties_with_froup_size(
                        neigbor[0], neigbor[1], env, initial_state)
                    score += (len(group_size) * SAVEMAGNITUDE) * self.additional_liberty_non_linearity(
                        len(neigbor_group_liberties), len(initial_group_liberties)
                    )

        return score * self.being_on_border_non_linearity(row, col)

    def get_group_non_linearity(self, number_of_liberties_left):
        if number_of_liberties_left > 2:
            return 0
        if number_of_liberties_left == 1:
            return 0.7
        if number_of_liberties_left == 2:
            return 0.2
        if number_of_liberties_left == 0:
            return 1

    def additional_liberty_non_linearity(self, new_liberties, initial_liberties):
        if 1 - new_liberties / initial_liberties > 0.5:
            return 1
        elif 1 - new_liberties / initial_liberties > 0.3:
            return 0.5
        elif 1 - new_liberties / initial_liberties > 0.25:
            return 0.15
        return 0.05

    def being_on_border_non_linearity(self, row, col):
        if row == 0:
            # 2 moves needed to block that group
            if col == 0 or col == BOARD_SIZE - 1:
                return 0.2
            # one move is needed
            if col > 0:
                return 0.45
        if row == BOARD_SIZE - 1:
            # 2 moves needed to block that group
            if col == 0 or col == BOARD_SIZE - 1:
                return 0.2
            if col > 0:
                return 0.45
        if col == 0:
            if row == 0 or row == BOARD_SIZE - 1:
                return 0.2
            if row > 0:
                return 0.45
        if col == BOARD_SIZE - 1:
            if row == 0 or row == BOARD_SIZE - 1:
                return 0.2
            if row > 0:
                return 0.45
        return 1

    def number_of_group_liberties_with_froup_size(self, row, col, env, altboard=None):
        adjacent_allies = env.find_allies(row, col, env.player_label, env.current_state)
        if altboard is None:
            altboard = env.current_copy()
            altboard[row][col] = env.player_label
        liberties = set()
        group = set()

        for ally in adjacent_allies:
            group.add(ally)
            neighbors = env.get_all_neighbors(ally[0], ally[1])
            for neighbor in neighbors:
                if altboard[neighbor[0]][neighbor[1]] == 0:
                    if neighbor not in liberties:
                        liberties.add(neighbor)
                elif altboard[neighbor[0]][neighbor[1]] == env.player_label:
                    group.add(neighbor)
        return liberties, group

    def number_of_enemy_group_liberties_with_group_size(self, row, col, env, altboard=None):
        adjacent_allies = env.find_allies(row, col, env.opponent_label, env.current_state)
        if altboard is None:
            altboard = env.current_copy()
            altboard[row][col] = env.player_label
        liberties = set()
        group = set()

        for ally in adjacent_allies:
            group.add(ally)
            neighbors = env.get_all_neighbors(ally[0], ally[1])
            for neighbor in neighbors:
                if altboard[neighbor[0]][neighbor[1]] == 0:
                    if neighbor not in liberties:
                        liberties.add(neighbor)
                elif altboard[neighbor[0]][neighbor[1]] == env.opponent_label:
                    group.add(neighbor)
        return liberties, group

    def close_up_init(self, row, col, env, nbs, dgs):
        all_neigbors = nbs + dgs
        my_nbs_chips = []
        my_diag_nbs = []
        enemy_neigbors = []

        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                my_nbs_chips.append(neigbor)
            elif env.current_state[nrow][ncol] == env.opponent_label:
                enemy_neigbors.append(neigbor)

        for neigbor in dgs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                my_diag_nbs.append(neigbor)

        if len(enemy_neigbors) > 0 and (len(my_nbs_chips) + len(my_diag_nbs) > 0):
            for enemy in enemy_neigbors:
                erow, ecol = enemy

                # move right
                if ecol > col:
                    if col < BOARD_SIZE - 1:
                        next_col = col + 1
                        while next_col <= BOARD_SIZE - 1:
                            if env.current_state[row][next_col] == env.player_label or \
                                    next_col == BOARD_SIZE - 1:
                                return (CLOSEUPVALUE + (
                                    abs(col - next_col)) * SIDEBREAKER) * self.closeup_on_border_nonlinearity(row, col)
                            elif env.current_state[row][next_col] == 0:
                                break
                            elif env.current_state[row][next_col] == env.opponent_label:
                                next_col += 1
                elif ecol < col:  # move left
                    if col > 0:
                        next_col = col - 1
                        while next_col >= 0:
                            if env.current_state[row][next_col] == env.player_label or \
                                    next_col == 0:
                                return (CLOSEUPVALUE + (
                                    abs(col - next_col)) * SIDEBREAKER) * self.closeup_on_border_nonlinearity(row, col)
                            elif env.current_state[row][next_col] == 0:
                                break
                            elif env.current_state[row][next_col] == env.opponent_label:
                                next_col -= 1
                elif erow > row:  # move down
                    if row < BOARD_SIZE - 1:
                        next_row = row + 1
                        while next_row <= BOARD_SIZE - 1:
                            if env.current_state[next_row][col] == env.player_label or \
                                    next_row == BOARD_SIZE - 1:
                                return (CLOSEUPVALUE + (
                                    abs(row - next_row)) * SIDEBREAKER) * self.closeup_on_border_nonlinearity(row, col)
                            elif env.current_state[next_row][col] == 0:
                                break
                            elif env.current_state[next_row][col] == env.opponent_label:
                                next_row += 1
                elif erow < row:  # move up
                    if row > 0:
                        next_row = row - 1
                        while next_row >= 0:
                            if env.current_state[next_row][col] == env.player_label or \
                                    next_row == 0:
                                return (CLOSEUPVALUE + (
                                    abs(row - next_row)) * SIDEBREAKER) * self.closeup_on_border_nonlinearity(row, col)
                            elif env.current_state[next_row][col] == 0:
                                break
                            elif env.current_state[next_row][col] == env.opponent_label:
                                next_row -= 1
        return 0

    def closeup_on_border_nonlinearity(self, row, col):
        if row == 0 or row == BOARD_SIZE - 1:
            if col == 0 or col == BOARD_SIZE - 1:
                return 0.55
            return 0.85
        if col == 0 or col == BOARD_SIZE - 1:
            return 0.85
        return 1

    def enclosed_init(self, row, col, env, nbs, dgs):
        all_neigbors = nbs + dgs
        my_nbs_chips = []
        enemy_diag_nbs = []
        enemy_neigbors = []

        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                my_nbs_chips.append(neigbor)
            elif env.current_state[nrow][ncol] == env.opponent_label:
                enemy_neigbors.append(neigbor)

        for neigbor in dgs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.opponent_label:
                enemy_diag_nbs.append(neigbor)

        if len(my_nbs_chips) > 0 and (len(enemy_neigbors) + len(enemy_diag_nbs) > 0):
            for ally in enemy_neigbors:
                arow, acol = ally

                # move right
                if acol > col:
                    if col < BOARD_SIZE - 1:
                        next_col = col + 1
                        while next_col <= BOARD_SIZE - 1:
                            if env.current_state[row][next_col] == env.opponent_label or \
                                    next_col == BOARD_SIZE - 1:
                                return -(CLOSEUPVALUE + (abs(col - next_col)) * SIDEBREAKER)
                            elif env.current_state[row][next_col] == 0:
                                break
                            elif env.current_state[row][next_col] == env.player_label:
                                next_col += 1
                elif acol < col:  # move left
                    if col > 0:
                        next_col = col - 1
                        while next_col >= 0:
                            if env.current_state[row][next_col] == env.opponent_label or \
                                    next_col == 0:
                                return -(CLOSEUPVALUE + (abs(col - next_col)) * SIDEBREAKER)
                            elif env.current_state[row][next_col] == 0:
                                break
                            elif env.current_state[row][next_col] == env.player_label:
                                next_col -= 1
                elif arow > row:  # move down
                    if row < BOARD_SIZE - 1:
                        next_row = row + 1
                        while next_row <= BOARD_SIZE - 1:
                            if env.current_state[next_row][col] == env.opponent_label or \
                                    next_row == BOARD_SIZE - 1:
                                return -(CLOSEUPVALUE + (abs(col - next_row)) * SIDEBREAKER)
                            elif env.current_state[next_row][col] == 0:
                                break
                            elif env.current_state[next_row][col] == env.player_label:
                                next_row += 1
                elif arow < row:  # move up
                    if row > 0:
                        next_row = row - 1
                        while next_row >= 0:
                            if env.current_state[next_row][col] == env.opponent_label or \
                                    next_row == 0:
                                return -(CLOSEUPVALUE + (abs(col - next_row)) * SIDEBREAKER)
                            elif env.current_state[next_row][col] == 0:
                                break
                            elif env.current_state[next_row][col] == env.player_label:
                                next_row -= 1
        return 0

    def number_of_group_liberties(self, row, col, env, altboard=None):
        adjacent_allies = env.find_allies(row, col, env.player_label, env.current_state)
        if altboard is None:
            altboard = env.current_copy()
            altboard[row][col] = env.player_label
        liberties = []

        for ally in adjacent_allies:
            neighbors = env.get_all_neighbors(ally[0], ally[1])
            for neighbor in neighbors:
                if altboard[neighbor[0]][neighbor[1]] == 0:
                    if neighbor not in liberties:
                        liberties.append(neighbor)
        return liberties

    def d_neigbors_by_sides(self, row, col, dgs, env):
        right_up = False
        left_up = False
        right_down = False
        left_down = False
        for neigbor in dgs:
            nrow, ncol = neigbor
            if ncol > col:
                if nrow > row:
                    if env.current_state[nrow][ncol] != 0:
                        right_down = neigbor
                elif nrow < row:
                    if env.current_state[nrow][ncol] != 0:
                        right_up = neigbor
            else:
                if nrow > row:
                    if env.current_state[nrow][ncol] != 0:
                        left_down = neigbor
                elif nrow < row:
                    if env.current_state[nrow][ncol] != 0:
                        left_up = neigbor

        return right_up, right_down, left_up, left_down

    def consume_grid_init(self, row, col, env, dgs):
        right_up, right_down, left_up, left_down = self.d_neigbors_by_sides(row, col, dgs, env)

        if right_up and right_down:
            if env.current_state[row][col + 1] == 0:
                return GRIDVALUE
        if right_down and left_down:
            if env.current_state[row + 1][col] == 0:
                return GRIDVALUE
        if left_down and left_up:
            if env.current_state[row][col - 1] == 0:
                return GRIDVALUE
        if left_up and right_up:
            if env.current_state[row - 1][col] == 0:
                return GRIDVALUE

        if right_up and col == 0:
            if env.current_state[row - 1][col] == 0 and \
                    env.current_state[row][col + 1] == 0:
                return GRIDVALUE
        if left_up and col == BOARD_SIZE - 1:
            if env.current_state[row - 1][col] == 0 and \
                    env.current_state[row][col - 1] == 0:
                return GRIDVALUE
        if right_down and col == 0:
            if env.current_state[row + 1][col] == 0:
                return GRIDVALUE
        if left_down and col == BOARD_SIZE - 1:
            if env.current_state[row + 1][col] == 0 and \
                    env.current_state[row][col - 1] == 0:
                return GRIDVALUE
        if left_down and row == 0:
            if env.current_state[row][col - 1] == 0 and \
                    env.current_state[row + 1][col] == 0 and \
                    env.current_state[row][col - 1] == 0:
                return GRIDVALUE
        if right_down and row == 0:
            if env.current_state[row][col + 1] == 0 and \
                    env.current_state[row + 1][col] == 0:
                return GRIDVALUE
        if left_up and row == BOARD_SIZE - 1:
            if env.current_state[row][col - 1] == 0 and \
                    env.current_state[row - 1][col]:
                return GRIDVALUE
        if right_up and row == GRIDVALUE:
            if env.current_state[row][col + 1] == 0 and \
                    env.current_state[row - 1][col]:
                return GRIDVALUE
        return 0

    def get_score_non_linearity(self, move_number):
        if move_number < 12:
            return 0.1
        else:
            return 0.1 * move_number

    def prev_move_non_linearity(self, row, col, env):
        distance = env.distance_to_move(row, col)
        if distance == 0:
            return 0
        if distance == 1 or distance == 3:
            return CLOSETOPREVMOVEVALUE / 2
        elif distance == 2:
            return CLOSETOPREVMOVEVALUE
        else:
            return CLOSETOPREVMOVEVALUE / distance

    def liberty_non_linearity(self, liberty_number):
        if liberty_number == 1:
            return 0
        elif liberty_number == 2:
            return 1.2
        elif liberty_number == 3:
            return 3
        elif liberty_number >= 4:
            return 3.4
        return 0

    def get_best_evaluated_move(self, environment, moveset, current_depth, parent_move, last_move=False,
                                evaluator=False):
        if moveset == []:
            self.history[current_depth][parent_move][ANYMOVE] = PASSSCORE
            return PASSSCORE, ANYMOVE
        if last_move == False:
            best_value = -np.inf
            move = None

            for row, col in moveset:
                dget = 0
                dsave = 0
                nbs = environment.get_all_neighbors(row, col)
                dgs = environment.get_diagonal_neigbors(row, col)

                liberties = self.number_of_group_liberties(row, col, environment)
                liberty_num = len(liberties)

                if len(liberties) <= 1 and not environment.is_potential_score(row, col):
                    move_eval = PROBABLEDEATHPENALTY
                else:
                    # GENERAL EVALUATION
                    move_eval = self.border_init(row, col, environment, nbs)

                    # GENERAL EVALUATION
                    if liberty_num < 5:
                        move_eval += self.liberty_non_linearity(liberty_num) * LIBERTYVALUE
                    else:
                        move_eval += 4 * LIBERTYVALUE

                    # ATTACKING EVALUATION
                    move_eval += self.close_pipe_init(row, col, environment, nbs)

                    # ATTACKING EVALUATION
                    move_eval += self.cut_init(row, col, environment, nbs, dgs)

                    if environment.n_move in [0, 1]:
                        if row == 2 and col == 2:
                            move_eval += 0.25

                    # GENERAL EVALUATION
                    move_eval += self.empty_row_init(row, environment)

                    # GENERAL EVALUATION
                    if environment.prev_move is not None:
                        move_eval += self.prev_move_non_linearity(row, col, environment)

                    if current_depth == 0:
                        dsave = move_eval

                    # SAVING EVALUATION
                    # Если противник будет пытаться сохранить фишки - нам не так страшно,
                    # как если он будет пытаться атаковать, по сему необходимо снижать баллы
                    # защитных ходов противника что бы остерегаться ходов атакующих.
                    if self.is_my_turn(current_depth):
                        move_eval += self.save_score_init(row, col, environment, current_depth)
                        move_eval += self.save_group_score(row, col, environment)
                    else:
                        move_eval += OPPONENTSSAVEEVALUATIONVALUE * self.save_score_init(row, col, environment,
                                                                                         current_depth)
                        move_eval += OPPONENTSSAVEEVALUATIONVALUE * self.save_group_score(row, col, environment)

                    # ---environment.player_label == 1 and
                    if current_depth == 0:
                        dsave = move_eval - dsave
                        dget = move_eval

                    get_score, is_rekillable_on_getting_score = self.get_score_init(row, col, environment,
                                                                                    current_depth)
                    move_eval += self.get_group_score(row, col, environment)
                    move_eval += get_score

                    if current_depth == 0:
                        dget = move_eval - dget
                        move_eval += (dsave + dget) * self.get_score_non_linearity(environment.n_move)

                    # ATTACKING EVALUATION
                    close_up_score = self.close_up_init(row, col, environment, nbs, dgs)
                    move_eval += close_up_score

                    # SAVEING ANALOGUE
                    if self.is_my_turn(current_depth):
                        move_eval += self.enclosed_init(row, col, environment, nbs, dgs)
                    else:
                        move_eval += self.enclosed_init(row, col, environment, nbs, dgs) * OPPONENTSSAVEEVALUATIONVALUE

                    is_closing_enemy = True if close_up_score > 0 else False

                    # GENERAL EVALUATION
                    move_eval += self.has_neigbor(row, col, environment, nbs, dgs)

                    # SEMI_ATTAKING EVALUATION
                    move_eval += self.breake_circle_init(row, col, environment, current_depth, parent_move, nbs,
                                                         dgs)
                    move_eval += (self.breake_circle_init(row, col, environment, current_depth, parent_move, nbs, dgs))

                    # ATTACKING EVALIUATION
                    move_eval += self.acquire_circle_init(row, col, environment, current_depth, parent_move, nbs, dgs)
                    move_eval += self.consume_grid_init(row, col, environment, dgs)

                    # GENERAL EVALUATION
                    if not is_closing_enemy:
                        move_eval += self.amongst_enemies_penalty(row, col, environment, nbs, dgs)

                    # SAVE EVALUATION
                    if self.is_my_turn(current_depth):
                        move_eval += self.amongts_all_allies(environment, nbs,
                                                             dgs) * self.taking_border_among_allies_penalty(row, col)
                    else:
                        move_eval += self.amongts_all_allies(environment, nbs,
                                                             dgs) * self.taking_border_among_allies_penalty(row,
                                                                                                            col) * OPPONENTSSAVEEVALUATIONVALUE

                    # SAVING EVALUATION
                    if self.is_my_turn(current_depth):
                        move_eval += self.provides_new_liberies(row, col, environment, nbs, dgs)
                    else:
                        move_eval += OPPONENTSSAVEEVALUATIONVALUE * (
                            self.provides_new_liberies(row, col, environment, nbs, dgs))

                    # DANGEROUS ENEMIES MOVES SHOULD BE ACCENTED
                    if not self.is_my_turn(current_depth):
                        distance = self.distance_to_move((row, col), environment.prev_opponents_move)
                        move_eval += POTENTIALTARGETSCORE / distance

                    # ATTACKING EVALUATION
                    move_eval += self.leave_one_liberty(row, col, environment, nbs)

                    # TODO: was 24/24 before that was added
                    move_eval += self.enemy_on_the_line_init(row, col, environment)

                    if not is_rekillable_on_getting_score:
                        if environment.get_winner_label() == environment.player_label:
                            move_eval *= 1.25

                    # FUTURE MOVES DISCARD
                    if current_depth > 1:
                        move_eval *= (FUTUREMOVE_DISCARD ** current_depth)
                # ____________________
                if evaluator == False:
                    if (row, col) not in self.history[current_depth][parent_move].keys():
                        self.history[current_depth][parent_move][(row, col)] = move_eval

                if move_eval > best_value:
                    best_value = move_eval
                    move = (row, col)
            return best_value, move
        else:
            if len(moveset) > 0:
                best_value = LOSER_SCORE
                best_move = random.choice(moveset)
                for row, col in moveset:
                    new_state_string = self.create_new_state_string(row, col, environment)
                    final_environment = Game(environment.N)
                    final_environment.init_from_state_string(new_state_string)
                    if final_environment.get_winner_label() == environment.player_label:
                        self.history[current_depth][parent_move][(row, col)] = WINNER_SCORE
                        best_value = WINNER_SCORE
                        best_move = (row, col)
                    else:
                        if (row, col) not in self.history[current_depth][parent_move].keys():
                            self.history[current_depth][parent_move][(row, col)] = LOSER_SCORE
                return best_value, best_move
            else:
                return 0, "PASS"

    def enemy_on_the_line_init(self, row, col, env):
        may_be_taken_left = False
        may_be_taken_righ = False
        may_be_taken_up = False
        may_be_taken_down = False

        if col < BOARD_SIZE - 1 and col > 0:
            if env.current_state[row][col - 1] == 0 and env.current_state[row][col + 1] == 0:
                may_be_taken_right = False
                next_col = col + 1
                while next_col <= BOARD_SIZE - 1:
                    if env.current_state[row][next_col] == env.opponent_label:
                        may_be_taken_right = True
                    if env.current_state[row][next_col] == env.player_label:
                        may_be_taken_right = False
                        break
                    next_col += 1
                prev_col = col - 1
        else:
            may_be_taken_right = True
        if col > 0 and col < BOARD_SIZE - 1:
            if env.current_state[row][col - 1] == 0 and env.current_state[row][col + 1] == 0:
                may_be_taken_left = False
                while prev_col >= 0:
                    if env.current_state[row][prev_col] == env.opponent_label:
                        may_be_taken_left = True
                    if env.current_state[row][prev_col] == env.player_label:
                        may_be_taken_left = False
                        break
                    prev_col -= 1
        else:
            may_be_taken_left = True
        if may_be_taken_left and may_be_taken_right:
            return 0.2

        if row < BOARD_SIZE - 1 and row > 0:
            if env.current_state[row + 1][col] == 0 or env.current_state[row - 1][col] == 0:
                may_be_taken_down = False
                next_row = row + 1
                while next_row <= BOARD_SIZE - 1:
                    if env.current_state[next_row][col] == env.opponent_label:
                        may_be_taken_down = True
                    if env.current_state[next_row][col] == env.player_label:
                        may_be_taken_down = False
                        break
                    next_row += 1
                prev_row = row - 1
        else:
            may_be_taken_down = True
        if row > 0 and row < BOARD_SIZE - 1:
            if env.current_state[row + 1][col] == 0 or env.current_state[row - 1][col] == 0:
                may_be_taken_up = False
                while prev_row >= 0:
                    if env.current_state[prev_row][col] == env.opponent_label:
                        may_be_taken_up = True
                    if env.current_state[prev_row][col] == env.player_label:
                        may_be_taken_up = False
                        break
                    prev_row -= 1
        else:
            may_be_taken_up = True
        if may_be_taken_up and may_be_taken_down:
            return 0.2
        return 0

    def amongst_enemies_penalty(self, row, col, env, nbs, dgs):
        all_neigbors = nbs
        vulnerable_liberty = self.vulnerability_magnitude(len(nbs))
        liberties = 0

        for neigbor in all_neigbors:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == 0:
                liberties += 1
            if env.current_state[nrow][ncol] == env.player_label or liberties > 2:
                return 0
        return AMONGENEMYPENALTY + vulnerable_liberty * AMONGENEMYPENALTY

    def taking_border_among_allies_penalty(self, row, col):
        if row == 0 or row == BOARD_SIZE - 1:
            if col == 0 or col == BOARD_SIZE - 1:
                return 4
            else:
                return 2
        if col == 0 or col == BOARD_SIZE - 1:
            return 2
        return 1

    def amongts_all_allies(self, env, nbs, dgs):
        adjacent_allies = 0
        diagonal_allies = 0

        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                adjacent_allies += 1
            if env.current_state[nrow][ncol] == env.opponent_label:
                return 0

        for neigbor in dgs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                diagonal_allies += 1

        return self.ally_neighboring_non_linearity(adjacent_allies, diagonal_allies, len(nbs),
                                                   len(dgs)) * AMONGALLIESPENALTY

    def ally_neighboring_non_linearity(self, adjacent_allies, diagonal_allies, max_nbs, max_dgs):
        if adjacent_allies / max_nbs == 1:
            if diagonal_allies / max_dgs in [0.75, 1]:
                return 10
            elif diagonal_allies / max_dgs in [0.25, 0.5]:
                return 8
            return 7
        if adjacent_allies / max_nbs == 0.75:
            if diagonal_allies / max_dgs in [0.75, 1]:
                return 8
            elif diagonal_allies / max_dgs in [0.25, 0.5]:
                return 5
            return 2
        if round(adjacent_allies / max_nbs, 1) == 0.6:
            if diagonal_allies / max_dgs == 0.5:
                return 5
            return 4
        if round(adjacent_allies / max_nbs, 1) == 0.3:
            if diagonal_allies / max_dgs == 0.5:
                return 4
            return 3
        if adjacent_allies / max_nbs == 0.5:
            if diagonal_allies / max_dgs in [0.75, 1]:
                return 3
            elif diagonal_allies / max_dgs in [0.25, 0.5]:
                return 2
            return 1
        return 0

    def preilous_break_position(self, env, nbs, dgs, opponents_check=False):
        all_neigbors = nbs + dgs
        player_count = 0
        enemy_count = 0
        for neigbor in all_neigbors:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.opponent_label:
                enemy_count += 1
            elif env.current_state[nrow][ncol] == env.player_label:
                player_count += 1
        if opponents_check:
            return enemy_count < player_count
        return enemy_count > player_count

    # ENEMY ONLY
    def leave_one_liberty(self, row, col, env, nbs):
        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.opponent_label:
                altboard = env.current_copy()
                altboard[row][col] == env.player_label
                left_liberties = self.count_opponents_liberties(nrow, ncol, env, altboard)
                if left_liberties == 1:
                    return POTENTIALTARGETSCORE
        return 0

    def count_opponents_liberties(self, row, col, env, altboard):
        neigbors = env.get_adjacent_allies(row, col, env.opponent_label, altboard)
        liberties = 0
        for neigbor in neigbors:
            nrow, ncol = neigbor
            if altboard[nrow][ncol] == 0:
                liberties += 1
        return liberties

    def breake_circle_init(self, row, col, env, current_depth, parent_move, move_nbs, move_dgs):
        finite_states = []
        adjacent_enemies = env.get_adjacent_allies(row, col, env.opponent_label, env.current_state)
        adjacent_dgn_enemies = env.get_adjacent_diagonal_allies(row, col, env.opponent_label, env.current_state)
        all_adjacent_enemies = adjacent_enemies + adjacent_dgn_enemies
        frontier = []

        for enemy in all_adjacent_enemies:
            frontier.append((enemy, (row, col)))

        frontier.append(((row, col), None))
        visited = []

        side_A = None
        side_B = None

        while frontier:
            chip_tuple = frontier.pop()
            chip = chip_tuple[0]
            chip_parent = chip_tuple[1]
            visited.append(chip_tuple)
            if chip != (row, col):  # the only not opponent chip in the set
                erow, ecol = chip
                adjacent_enemy_allies = env.get_adjacent_allies(erow, ecol, env.opponent_label, env.current_state)
                adjacent_dgn_enemies = env.get_adjacent_diagonal_allies(erow, ecol, env.opponent_label,
                                                                        env.current_state)
                all_adjacent_enemies = adjacent_enemy_allies + adjacent_dgn_enemies

            if env.is_cell_on_border(chip) and chip_tuple not in finite_states:
                if len(finite_states) == 0:
                    finite_states.append(chip_tuple)
                    side_A = self.get_path(visited, finite_states[0])
                elif len(finite_states) == 1:
                    if self.distance_to_move((chip[0], chip[1]), finite_states[0][0]) > 1:
                        finite_states.append(chip_tuple)
                        side_B = self.get_path(visited, finite_states[1])

                if chip_tuple not in visited:
                    visited.append(chip_tuple)
                if len(finite_states) >= 2:
                    altboard = env.current_copy()
                    path = side_A.union(side_B)
                    test_path = copy.deepcopy(path)
                    test_path.remove(finite_states[1][0])

                    if self.is_valid_path(test_path, env, finite_states[0][0], finite_point_2=None):
                        finite_states.remove(finite_states[1])
                        continue

                    path = self.trim_path(path, [finite_states[0][0], finite_states[1][0]])

                    if not self.is_valid_path(path, env, finite_states[0][0], finite_states[1][0]):
                        finite_states.remove(finite_states[1])
                        continue

                    altboard[row][col] = env.opponent_label
                    area = self.find_circled_area(finite_states[0][0], finite_states[1][0], path, env, True)

                    if len(area) > 0:
                        zeroes, points = self.differetiate_area(area, env, True)
                        circle_value = CIRCLEVALUE + (self.area_non_linearity(zeroes, points, env.n_move,
                                                                              env.max_move) * CIRCLEDEPTHDISCARD ** current_depth)
                        if self.preilous_break_position(env, move_nbs, move_dgs, opponents_check=True):
                            circle_value *= PERILOUSPOSITIONDISCARD
                        return circle_value
                    else:
                        finite_states.remove(finite_states[1])

            for enemy in all_adjacent_enemies:
                if (enemy, chip) not in frontier and (enemy, chip) not in visited:
                    frontier.append((enemy, chip))
        return 0

    def vulnerability_magnitude(self, liberty_num):
        if liberty_num == 2:
            return 0.25
        elif liberty_num == 3:
            return 0.15
        else:
            return 0

    def area_non_linearity(self, zeroes, points, move_number, max_move):
        move_range = (max_move - move_number)
        points_value = (CIRCLEMAGNITUDE) * points
        zeroes_value = ((CIRCLEMAGNITUDE / 2) * zeroes) / move_range
        return self.area_magnitude(zeroes + points) * (zeroes_value + points_value)

    def area_magnitude(self, square):
        if square == 1:
            return 0.3
        elif square == 2 or square == 3:
            return 0.8
        elif square == 4:
            return 0.4
        elif square > 4:
            return 0.1
        else:
            return 0

    def is_no_area(self, path):
        vertex_1 = path[0][0]
        vertex_2 = path[1][0]

    def get_path(self, visited, finite_point):
        path = set()
        path.add(finite_point[0])
        vertex = finite_point[1]
        while vertex != None:
            path.add(vertex)
            vertex = self.get_parent(visited, vertex)
        return path

    def get_parent(self, visited, vertex):
        for point in visited:
            if point[0] == vertex:
                return point[1]
        visited.remove(vertex)

    def acquire_circle_init(self, row, col, env, current_depth, parent_move, move_nbs, move_dgs):
        finite_states = []
        adjacent_allies = env.get_adjacent_allies(row, col, env.player_label, env.current_state)
        adjacent_dgn_allies = env.get_adjacent_diagonal_allies(row, col, env.player_label, env.current_state)
        all_adjacent_allies = adjacent_allies + adjacent_dgn_allies
        frontier = []

        for ally in all_adjacent_allies:
            frontier.append((ally, (row, col)))

        frontier.append(((row, col), None))
        visited = []

        side_A = None
        side_B = None

        while frontier:
            chip_tuple = frontier.pop()
            chip = chip_tuple[0]
            chip_parent = chip_tuple[1]
            visited.append(chip_tuple)
            if chip != (row, col):  # if not consider it then those on border wont be counted!!!
                erow, ecol = chip
                adjacent_my_allies = env.get_adjacent_allies(erow, ecol, env.player_label, env.current_state)
                adjacent_dgn_allies = env.get_adjacent_diagonal_allies(erow, ecol, env.player_label, env.current_state)
                all_adjacent_allies = adjacent_my_allies + adjacent_dgn_allies

            if env.is_cell_on_border(chip) and chip_tuple not in finite_states:
                if len(finite_states) == 0:
                    finite_states.append(chip_tuple)
                    side_A = self.get_path(visited, finite_states[0])
                elif len(finite_states) == 1:
                    if self.distance_to_move((chip[0], chip[1]), finite_states[0][0]) > 1:
                        finite_states.append(chip_tuple)
                        side_B = self.get_path(visited, finite_states[1])

                if chip_tuple not in visited:
                    visited.append(chip_tuple)
                if len(finite_states) >= 2:
                    altboard = env.current_copy()
                    path = side_A.union(side_B)
                    test_path = copy.deepcopy(path)
                    test_path.remove(finite_states[1][0])

                    if self.is_valid_path(test_path, env, finite_states[0][0], finite_point_2=None):
                        finite_states.remove(finite_states[1])
                        continue

                    path = self.trim_path(path, [finite_states[0][0], finite_states[1][0]])

                    # ПРодолжить поиск
                    if not self.is_valid_path(path, env, finite_states[0][0], finite_states[1][0]):
                        finite_states.remove(finite_states[1])
                        continue

                    altboard[row][col] = env.player_label
                    area = self.find_circled_area(finite_states[0][0], finite_states[1][0], path, env, False)

                    if len(area) > 0:
                        zeroes, points = self.differetiate_area(area, env, False)
                        circle_value = CIRCLEVALUE + (self.area_non_linearity(zeroes, points, env.n_move,
                                                                              env.max_move) * CIRCLEDEPTHDISCARD ** current_depth)
                        if self.preilous_break_position(env, move_nbs, move_dgs):
                            circle_value *= PERILOUSPOSITIONDISCARD
                        return circle_value
                    else:
                        finite_states.remove(finite_states[1])

            for ally in all_adjacent_allies:
                if (ally, chip) not in frontier and (ally, chip) not in visited:
                    frontier.append((ally, chip))
        return 0

    def differetiate_area(self, area, env, is_breaking):
        zeroes = 0
        points = 0
        if is_breaking:
            for cell in area:
                crow, ccol = cell
                if env.current_state[crow][ccol] == 0:
                    zeroes += 1
                elif env.current_state[crow][ccol] == env.player_label:
                    points += 1
        else:
            for cell in area:
                crow, ccol = cell
                if env.current_state[crow][ccol] == 0:
                    zeroes += 1
                elif env.current_state[crow][ccol] == env.opponent_label:
                    points += 1
        return points, zeroes

    def is_valid_path(self, trimmed_path, env, finite_point_1, finite_point_2):
        frontier = [finite_point_1]
        visited = []
        if finite_point_2 is None:
            while frontier:
                chip = frontier.pop()
                crow, ccol = chip
                nbs = env.get_all_neighbors(crow, ccol)
                dgs = env.get_diagonal_neigbors(crow, ccol)
                all_neigbors = nbs + dgs
                visited.append(chip)

                if chip != finite_point_1 and (chip[0] == 0 or chip[0] == BOARD_SIZE - 1 or \
                                               chip[1] == 0 or chip[1] == BOARD_SIZE - 1):
                    return True

                for neigbor in all_neigbors:
                    if neigbor in trimmed_path:
                        if neigbor not in frontier and neigbor not in visited:
                            frontier.append(neigbor)
            return False
        else:
            while frontier:
                chip = frontier.pop()
                crow, ccol = chip
                nbs = env.get_all_neighbors(crow, ccol)
                dgs = env.get_diagonal_neigbors(crow, ccol)
                all_neigbors = nbs + dgs
                visited.append(chip)

                if chip == finite_point_2:
                    return True

                for neigbor in all_neigbors:
                    if neigbor in trimmed_path:
                        if neigbor not in frontier and neigbor not in visited:
                            frontier.append(neigbor)
            return False

    # RULES for patterns such as:
    #
    #     X                        X
    #   X   X         XXX        O   X
    #     X           XOX          X
    #                  X
    #
    def close_pipe_init(self, row, col, env, nbs):
        right_up = (row - 1, col + 1) if row > 0 and col < BOARD_SIZE - 1 else (False, False)
        right_next = (row, col + 1) if col < BOARD_SIZE - 1 else (False, False)
        right_bot = (row + 1, col + 1) if col < BOARD_SIZE - 1 and row < BOARD_SIZE - 1 else (False, False)

        down_next = (row + 1, col) if row < BOARD_SIZE - 1 else (False, False)
        up_next = (row - 1, col) if row > 0 else (False, False)

        left_up = (row - 1, col - 1) if row > 0 and col > 0 else (False, False)
        left_next = (row, col - 1) if col > 0 else (False, False)
        left_bot = (row + 1, col - 1) if row < BOARD_SIZE - 1 and col > 0 else (False, False)

        number_of_adjacent_enemies = 0
        number_of_adjacent_allies = 0
        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                number_of_adjacent_allies += 1
            elif env.current_state[nrow][ncol] == env.opponent_label:
                number_of_adjacent_enemies += 1

        if left_bot or right_bot or left_up or left_bot:
            if number_of_adjacent_enemies in [1, 2] and number_of_adjacent_allies in [0, 1]:
                return CLOSEPIPEVALUE

        if col == 0:
            # check up
            if row > 0:
                if right_up and up_next:
                    if env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # check right
                if right_up and right_next and (right_bot or row == BOARD_SIZE - 1):
                    if env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            (env.current_state[right_bot[0]][right_bot[1]] == env.player_label or \
                             row == BOARD_SIZE - 1):
                        return CLOSEPIPEVALUE
                # check directly up
                if right_next and up_next and right_up:
                    if env.current_state[right_next[0]][right_next[1]] == env.player_label and \
                            env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # check directly down
                if down_next and right_next and right_bot:
                    if env.current_state[right_next[0]][right_next[1]] == env.player_label and \
                            env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE

                if right_bot and down_next and right_next:
                    if env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            env.current_state[down_next[0]][down_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == 0 and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if right_up and up_next and right_next:
                    if env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            env.current_state[up_next[0]][up_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == 0 and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
            # check down
            if row == 0:
                if right_bot and down_next:
                    if env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # check right
                if right_bot and right_next:
                    if env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if right_bot and right_next and down_next:
                    if env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            env.current_state[down_next[0]][down_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == 0 and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
        if col > 0:
            if row > 0:
                # check left
                if left_up and left_next and (left_bot or row == BOARD_SIZE - 1):
                    if env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                            (env.current_state[left_bot[0]][left_bot[1]] == env.player_label or \
                             row == BOARD_SIZE - 1):
                        return CLOSEPIPEVALUE
                # check right
                if right_up and right_next and (right_bot or row == BOARD_SIZE - 1):
                    if env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[0]] == env.opponent_label and \
                            (env.current_state[right_bot[0]][right_bot[1]] == env.player_label or \
                             row == BOARD_SIZE - 1):
                        return CLOSEPIPEVALUE
                # check up
                if left_up and up_next and (right_up or col == BOARD_SIZE - 1):
                    if env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label and \
                            (env.current_state[right_up[0]][right_up[1]] == env.player_label or \
                             col == BOARD_SIZE - 1):
                        return CLOSEPIPEVALUE
                # check down
                if left_bot and down_next and (right_bot or col == BOARD_SIZE - 1):
                    if env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label and \
                            (env.current_state[right_bot[0]][right_bot[1]] == env.player_label or \
                             col == BOARD_SIZE - 1):
                        return CLOSEPIPEVALUE
                # directly up
                if left_next and right_next and left_up and right_up and up_next:
                    if env.current_state[left_next[0]][left_next[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.player_label and \
                            env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                            env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # directly righ
                if up_next and down_next and right_up and right_bot and right_next:
                    if env.current_state[up_next[0]][up_next[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.player_label and \
                            env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # directly down
                if left_next and right_next and left_bot and right_bot and down_next:
                    if env.current_state[left_next[0]][left_next[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.player_label and \
                            env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # directly left
                if up_next and down_next and left_bot and left_up and left_next:
                    if env.current_state[up_next[0]][up_next[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.player_label and \
                            env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE

                if left_up and up_next and left_next:
                    if env.current_state[left_up[0]][left_next[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                            env.current_state[up_next[0]][up_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[left_up[0]][left_next[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == 0 and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if right_up and right_next and up_next:
                    if env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            env.current_state[up_next[0]][up_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[right_up[0]][right_up[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == 0 and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if right_bot and right_next and down_next:
                    if env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == env.opponent_label and \
                            env.current_state[down_next[0]][down_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[right_bot[0]][right_bot[1]] == env.player_label and \
                            env.current_state[right_next[0]][right_next[1]] == 0 and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if left_bot and left_next and down_next:
                    if env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                            env.current_state[down_next[0]][down_next[1]] == 0:
                        return CLOSEPIPEVALUE
                    elif env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[left_next[0]][left_next[1]] == 0 and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
            if col == BOARD_SIZE - 1:
                if left_next and up_next and left_up:
                    if env.current_state[left_next[0]][left_next[1]] == env.player_label and \
                            env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                            env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                # check directly down
                if down_next and left_next and left_bot:
                    if env.current_state[left_next[0]][left_next[1]] == env.player_label and \
                            env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                            env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                        return CLOSEPIPEVALUE
                if row > 0:
                    if left_up and left_next and up_next:
                        if env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                                env.current_state[up_next[0]][up_next[1]] == 0:
                            return CLOSEPIPEVALUE
                        elif env.current_state[left_up[0]][left_up[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == 0 and \
                                env.current_state[up_next[0]][up_next[1]] == env.opponent_label:
                            return CLOSEPIPEVALUE
                    if left_bot and left_next and down_next:
                        if env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                                env.current_state[down_next[0]][down_next[1]] == 0:
                            return CLOSEPIPEVALUE
                        elif env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == 0 and \
                                env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                            return CLOSEPIPEVALUE
                if row == 0:
                    if left_bot and left_next and down_next:
                        if env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == env.opponent_label and \
                                env.current_state[down_next[0]][down_next[1]] == 0:
                            return CLOSEPIPEVALUE
                        elif env.current_state[left_bot[0]][left_bot[1]] == env.player_label and \
                                env.current_state[left_next[0]][left_next[1]] == 0 and \
                                env.current_state[down_next[0]][down_next[1]] == env.opponent_label:
                            return CLOSEPIPEVALUE

        return 0

    def trim_path(self, path, finite_points):
        trimmed_path = set()
        trimmed_path.add(finite_points[0])
        trimmed_path.add(finite_points[1])
        if self.path_can_be_reduced(finite_points[0], finite_points[1]):
            return trimmed_path
        for point in path:
            if point not in finite_points:
                prow, pcol = point
                if (prow != 0 and prow != BOARD_SIZE - 1) and (pcol != 0 and pcol != BOARD_SIZE - 1):
                    trimmed_path.add(point)
        return trimmed_path

    def path_can_be_reduced(self, finitePoint_1, finite_point_2):
        nbs = Game(BOARD_SIZE).get_all_neighbors(finitePoint_1[0], finitePoint_1[1])
        dgs = Game(BOARD_SIZE).get_diagonal_neigbors(finitePoint_1[0], finitePoint_1[1])
        all_neigbors = nbs + dgs
        if finite_point_2 in all_neigbors:
            return True
        return False

    def reduce_circles(self, max_depth, player_circle_dictionary):
        for layer in range(0, max_depth):  # 0
            for move in player_circle_dictionary[layer]:  # (-1,-1)
                for circle in player_circle_dictionary[layer][move]:
                    current_circle_path = set(circle[1])
                    move_that_generated_current_circle = circle[0]
                    next_circles = player_circle_dictionary[layer + 1][move_that_generated_current_circle]

                    for next_circle in next_circles:
                        next_circle_path = set(next_circle[1])
                        next_circle_score = next_circle[2]
                        next_circle_move = next_circle[0]
                        if len(current_circle_path.intersection(next_circle_path)) > 1:
                            now_score = self.history[layer + 1][move_that_generated_current_circle][next_circle_move]
                            self.history[layer + 1][move_that_generated_current_circle][
                                next_circle_move] -= next_circle_score

    def is_a_circle_subset(self, circle_dictionary, finite_points, current_depth):
        previous_depth = current_depth - 1
        finite_points_set = set(finite_points)
        if previous_depth not in circle_dictionary.keys():
            return False
        else:
            for existing_path in circle_dictionary[previous_depth]:
                existing_path_set = set(existing_path)
                if finite_points_set.issubset(existing_path_set):
                    return True
            return False

    def find_circled_area(self, vertex_a, vertex_b, path, env, is_breaking=True):
        area = set()
        if vertex_a[0] == vertex_b[0] == 0:
            # upper bound
            area = self.upper_bound_area(path, env, is_breaking)
        elif vertex_a[0] == vertex_b[0] == (BOARD_SIZE - 1):
            # bottom bound
            area = self.bottom_bound_area(path, env, is_breaking)
        else:
            # either bottom or upper chose is - min since take over smaller region is easier
            side, alt = self.get_side_and_alt_vertices(path)
            axes = None
            if alt is None:
                alt, axes = self.get_alt_on_same_axes(path, side)
            elif side is None:
                side, axes = self.get_alt_on_same_axes(path, alt)
            if axes == 'x':
                area_a = self.skewed_area(path, env, is_breaking, left=None, up=True)
                area_b = self.skewed_area(path, env, is_breaking, left=None, up=False)
            else:
                area_a = self.skewed_area(path, env, is_breaking, left=True, up=None)
                area_b = self.skewed_area(path, env, is_breaking, left=False, up=None)
            if len(area_a) < len(area_b):
                return area_a
            else:
                return area_b

            if side[0] < alt[0] and side[1] < alt[1]:
                area = self.skewed_area(path, env, is_breaking, left=True, up=False)
            elif side[0] > alt[0] and side[1] < alt[1]:
                area = self.skewed_area(path, env, is_breaking, left=True, up=True)
            elif side[0] < alt[0] and side[1] > alt[1]:
                area = self.skewed_area(path, env, is_breaking, left=False, up=False)
            elif side[0] > alt[0] and side[1] > alt[1]:
                area = self.skewed_area(path, env, is_breaking, left=False, up=True)
        return area

    def get_side_and_alt_vertices(self, path):
        side = None
        alt = None
        for vertex in path:
            vrow, vcol = vertex
            if vcol == 0 or vcol == BOARD_SIZE - 1:
                side = (vrow, vcol)
            elif vrow == 0 or vrow == BOARD_SIZE - 1:
                alt = (vrow, vcol)

        return side, alt

    def get_alt_on_same_axes(self, path, side):
        side = side
        alt = None
        for vertex in path:
            if vertex != side:
                vrow, vcol = vertex
                srow, scol = side
                if srow == vrow:
                    if vrow == 0 or vrow == BOARD_SIZE - 1:
                        return (vrow, vcol), 'x'
                    if vcol == 0 or vcol == BOARD_SIZE - 1:
                        return (vrow, vcol), 'x'
                if scol == vcol:
                    if vcol == 0 or vcol == BOARD_SIZE - 1:
                        return (vrow, vcol), 'y'
                    if vrow == 0 or vrow == BOARD_SIZE - 1:
                        return (vrow, vcol), 'y'
                if vrow == 0 or vrow == BOARD_SIZE - 1:
                    return (vrow, vcol), 'y'
                if vcol == 0 or vcol == BOARD_SIZE - 1:
                    return (vrow, vcol), 'x'

    def upper_bound_area(self, path, env, is_breaking=True):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            upper_row = vrow - 1
            while upper_row >= 0:
                if (upper_row, vcol) not in path:
                    if is_breaking:
                        if env.current_state[upper_row][vcol] == 0 or env.current_state[upper_row][
                            vcol] == env.player_label:
                            area.add((upper_row, vcol))
                    else:
                        if env.current_state[upper_row][vcol] == 0 or env.current_state[upper_row][
                            vcol] == env.opponent_label:
                            area.add((upper_row, vcol))
                upper_row -= 1
        return area

    def bottom_bound_area(self, path, env, is_breaking=True):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            down_row = vrow + 1
            while down_row <= (BOARD_SIZE - 1):
                if (down_row, vcol) not in path:
                    if is_breaking:
                        if env.current_state[down_row][vcol] == 0 or env.current_state[down_row][
                            vcol] == env.player_label:
                            area.add((down_row, vcol))
                    else:
                        if env.current_state[down_row][vcol] == 0 or env.current_state[down_row][
                            vcol] == env.opponent_label:
                            area.add((down_row, vcol))
                down_row += 1
        return area

    def skewed_area(self, path, env, is_breaking, left, up):
        area = set()
        for vertex in path:
            vrow, vcol = vertex

            if left == False:
                right_col = vcol + 1
                while right_col <= (BOARD_SIZE - 1):
                    if (vrow, right_col) not in path and self.is_righter_than_path(vrow, right_col, path):
                        if is_breaking:
                            if env.current_state[vrow][right_col] == 0 or env.current_state[vrow][
                                right_col] == env.player_label:
                                area.add((vrow, right_col))
                        else:
                            if env.current_state[vrow][right_col] == 0 or env.current_state[vrow][
                                right_col] == env.opponent_label:
                                area.add((vrow, right_col))
                    right_col += 1

            if up == False:
                # down cycle
                down_row = vrow + 1
                while down_row <= (BOARD_SIZE - 1):
                    if (down_row, vcol) not in path and self.is_lower_than_path(down_row, vcol, path):
                        if is_breaking:
                            if env.current_state[down_row][vcol] == 0 or env.current_state[down_row][
                                vcol] == env.player_label:
                                area.add((down_row, vcol))
                        else:
                            if env.current_state[down_row][vcol] == 0 or env.current_state[down_row][
                                vcol] == env.opponent_label:
                                area.add((down_row, vcol))
                    down_row += 1

            if left == True:
                left_col = vcol - 1
                while left_col >= 0:
                    if (vrow, left_col) not in path and self.is_lefter_than_path(vrow, left_col, path):
                        if is_breaking:
                            if env.current_state[vrow][left_col] == 0 or env.current_state[vrow][
                                left_col] == env.player_label:
                                area.add((vrow, left_col))
                        else:
                            if env.current_state[vrow][left_col] == 0 or env.current_state[vrow][
                                left_col] == env.opponent_label:
                                area.add((vrow, left_col))
                    left_col -= 1

            if up == True:
                upper_row = vrow - 1
                while upper_row >= 0:
                    if (upper_row, vcol) not in path and self.is_upper_than_path(upper_row, vcol, path):
                        if is_breaking:
                            if env.current_state[upper_row][vcol] == 0 or env.current_state[upper_row][
                                vcol] == env.player_label:
                                area.add((upper_row, vcol))
                        else:
                            if env.current_state[upper_row][vcol] == 0 or env.current_state[upper_row][
                                vcol] == env.opponent_label:
                                area.add((upper_row, vcol))
                    upper_row -= 1
        return area

    def is_upper_than_path(self, row, col, path):
        for vertex in path:
            vrow, vcol = vertex
            if col == vcol:
                if row > vrow:
                    return False
        return True

    def is_lower_than_path(self, row, col, path):
        for vertex in path:
            vrow, vcol = vertex
            if col == vcol:
                if row < vrow:
                    return False
        return True

    def is_lefter_than_path(self, row, col, path):
        for vertex in path:
            vrow, vcol = vertex
            if row == vrow:
                if col > vcol:
                    return False
        return True

    def is_righter_than_path(self, row, col, path):
        for vertex in path:
            vrow, vcol = vertex
            if row == vrow:
                if col < vcol:
                    return False
        return True

    # NBS - parameter
    def has_neigbor(self, row, col, env, nbs, dgs):
        my_chips = 0
        my_dgs = 0
        for neigbor in nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                my_chips += 2
                break

        for neigbor in dgs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                my_dgs += 2
                break

        return my_chips * HASNEIGBOR + my_dgs * (HASNEIGBOR / 2)

    def quadrants_init(self, move_row, move_col, env):
        quadrants = [(1, 1), (1, 3), (3, 1), (3, 3)]

        if (move_row, move_col) in quadrants:
            neigbors = env.get_all_neighbors(move_row, move_col)
            diag_neigbors = env.get_diagonal_neigbors(move_row, move_col)

            if len(neigbors) + len(diag_neigbors) == 8:
                return QUADRANTVALUE
        return 0

    def border_init(self, row, col, env, nbs):
        if len(nbs) == 2:
            return INITVALUE / 1.5
        elif len(nbs) == 3:
            return INITVALUE / 1.9
        return INITVALUE

    def distance_fitness(self, distance):
        if distance == 1 or 3:
            return GROUPVALUE
        elif distance == 2:
            return 2 * GROUPVALUE
        else:
            return 0

    def group_move_init(self, row, col, env):
        distance = env.distance_to_move(row, col)
        if distance == 1:
            if len(env.get_all_neighbors(row, col)) < 4:
                return self.distance_fitness(distance) + BORDERVALUE
            return self.distance_fitness(distance)
        elif distance == 2:
            if len(env.get_all_neighbors(row, col)) < 4:
                return self.distance_fitness(distance) + BORDERVALUE
            return self.distance_fitness(distance)
        return self.distance_fitness(distance)

    def provides_new_liberies(self, row, col, env, nbs, dgs):
        all_nbs = nbs + dgs
        zero_neigbors = 0
        for neigbor in all_nbs:
            nrow, ncol = neigbor
            if env.current_state[nrow][ncol] == env.player_label:
                for adjacent_negbor in nbs:
                    arow, acol = adjacent_negbor
                    if env.current_state[arow][acol] == 0:
                        zero_neigbors += 1
                return zero_neigbors * LIBERTYVALUE
        return 0

    def empty_row_init(self, row, env):
        row_is_empty = True
        for col in range(BOARD_SIZE):
            if env.current_state[row][col] == env.player_label:
                row_is_empty = False

        if row_is_empty:
            return TAKINGOVERAROWSCORE
        return 0

    def cut_init(self, row, col, env, nbs, dgs):
        my_chips = 0
        neutral_chips = 0
        enemy_chips = 0

        my_diag_chips = 0
        neutral_diag_chips = 0
        enemy_diag_chips = 0

        enemies_on_both_sides, enemies_no_both_altitudes = self.enemies_on_sides(row, col, nbs, env)

        for n_row, n_col in nbs:
            chip = env.current_state[n_row][n_col]
            if chip == env.player_label:
                my_chips += 1
            elif chip == 0:
                neutral_chips += 1
            else:
                enemy_chips += 1

        for d_row, d_col in dgs:
            chip = env.current_state[d_row][d_col]
            if chip == env.player_label:
                my_diag_chips += 1
            elif chip == 0:
                neutral_diag_chips += 1
            else:
                enemy_diag_chips += 1

        if my_diag_chips >= 1 and enemy_diag_chips == 0 and enemy_chips == 0:
            return RINGVALUE
        elif my_diag_chips >= 1 and enemy_chips in [1, 2] and (enemy_chips + enemy_diag_chips) <= 2:
            return RINGVALUE
        elif enemy_diag_chips == 3 and neutral_diag_chips == 3 and neutral_chips == 3:
            return RINGVALUE
        elif enemy_diag_chips == 3 and my_diag_chips == 1:
            if neutral_chips == 4 or neutral_chips == 3:
                return RINGVALUE
            else:
                return RINGVALUE / 4
        elif enemy_diag_chips == 2 and neutral_chips == 3 and (
                neutral_diag_chips + my_diag_chips + enemy_diag_chips) == 2:
            return RINGVALUE
        elif my_diag_chips == 2 and neutral_chips == 3 and (neutral_diag_chips + my_diag_chips + enemy_diag_chips) == 2:
            return RINGVALUE
        elif len(nbs) == 3 and my_chips == 1 and enemy_diag_chips >= 1 and enemy_chips <= 1:
            return RINGVALUE
        elif len(nbs) == 3 and enemy_chips == 1 and my_diag_chips >= 1:
            return RINGVALUE
        elif enemy_diag_chips == 1 and my_diag_chips == 1 and neutral_chips == 4:
            return RINGVALUE
        elif enemy_diag_chips == 1 and enemy_chips == 1 and my_diag_chips == 0 and my_chips == 0:
            e_diag = dgs[0]
            e_nb = nbs[0]
            if e_diag[1] == e_nb[1] or e_diag[0] == e_nb[0]:
                return RINGVALUE / 4
            return RINGVALUE
        elif enemy_diag_chips == 1 and enemy_chips == 1 and my_diag_chips == 1:
            return RINGVALUE
        elif enemy_diag_chips == 1 and my_diag_chips == 1 and my_chips == 1:
            return RINGVALUE
        elif my_chips == 1 and my_diag_chips == 1 and enemy_chips in [1, 2]:
            return HALFDIAMONDVALUE
        elif enemy_diag_chips == 2 and my_diag_chips == 2:
            if neutral_chips >= 2:
                return DIAMONDVALUE
            elif neutral_chips == 1:
                return DIAMONDVALUE / 4
        elif my_diag_chips == 2:
            if neutral_diag_chips + neutral_chips == 6:
                return DIAMONDVALUE * 2
            if neutral_diag_chips + neutral_chips >= 4:
                return DIAMONDVALUE
            else:
                return HALFDIAMONDVALUE
        elif enemy_diag_chips == 2 and neutral_diag_chips == 2:
            if enemy_chips <= 2:
                return HALFDIAMONDVALUE
            else:
                return HALFDIAMONDVALUE / 4
        elif enemies_on_both_sides and not enemies_no_both_altitudes:
            return SIDEBREAKER
        elif enemies_no_both_altitudes and not enemies_on_both_sides:
            return SIDEBREAKER
        elif enemy_chips == 1 and my_diag_chips > 0:
            return SIDEBREAKER
        elif enemy_chips == 2 and (not enemies_on_both_sides and not enemies_no_both_altitudes):
            if neutral_chips + neutral_diag_chips == 6:
                return SINGLEENEMYVALUE * 2
            else:
                return SINGLEENEMYVALUE
        elif enemy_chips == 1 and (my_chips >= 2):
            return SINGLEENEMYVALUE * 2
        elif enemy_chips == 1:
            return SINGLEENEMYVALUE
        return 0

    def enemies_on_sides(self, row, col, nbs, env):
        left = (row, col - 1)
        right = (row, col + 1)

        up = (row - 1, col)
        bottom = (row + 1, col)

        sides = False
        alts = False

        if left in nbs and right in nbs:
            lr, lc = left
            rr, rc = right
            if env.current_state[lr][lc] == env.opponent_label and env.current_state[rr][rc] == env.opponent_label:
                sides = True

        if up in nbs and bottom in nbs:
            ur, uc = up
            br, bc = bottom
            if env.current_state[ur][uc] == env.opponent_label and env.current_state[br][bc] == env.opponent_label:
                alts = True

        return sides, alts

    def save_score_init(self, row, col, env, depth):
        loss = env.is_potential_loss(row, col)
        if loss:
            if env.is_move_valid(row, col, opponent_check=True):
                return SCOREBASE + loss * (SAVEMAGNITUDE ** (depth + 1))
        return 0

    def get_score_init(self, row, col, env, depth):
        score = env.is_potential_score(row, col)
        power = 1
        if depth not in [0, 1]:
            power = depth + 1
        if score:
            # Need double check here to veryfy that next move wont be worse
            if len(score) == 1:
                if self.is_rekillable(row, col, env, score[0]):
                    return (SCOREBASE + len(score) * (WIPE_MAGNITUDE ** power)) * 0.1, True
            return SCOREBASE + len(score) * (WIPE_MAGNITUDE ** power), False
        return 0, False

    def is_rekillable(self, row, col, environment, wipe_chips):
        new_envitonment = copy.deepcopy(environment)
        new_envitonment.current_state[row][col] = environment.player_label
        new_envitonment.wipe([wipe_chips], new_envitonment.current_state)

        opponent_environment = copy.deepcopy(new_envitonment)
        opponent_environment.swap_players()
        opponent_environment.current_state[wipe_chips[0]][wipe_chips[1]] = opponent_environment.player_label
        opponent_score = opponent_environment.is_potential_score(wipe_chips[0], wipe_chips[1])
        if not opponent_score:
            return False
        else:
            opponent_environment.wipe(opponent_score, opponent_environment.current_state)
            if environment.are_states_identical(environment.current_state, opponent_environment.current_state):
                return False
            else:
                return True

    def is_my_turn(self, depth):
        if depth % 2 == 0: return True
        return False

    def get_child_score(self, layer, parent):
        if layer == self.max_depth:
            best_value = -np.inf
            best_move = None
            for child in self.history[layer][parent]:
                if self.history[layer][parent][child] > best_value:
                    best_value = self.history[layer][parent][child]
                    best_move = child
            if self.is_my_turn(layer): return best_value, best_move
            return best_value, best_move
        else:
            best_value = -np.inf
            best_move = None
            for child in self.history[layer][parent]:
                expected_val, next_move = self.get_child_score(layer + 1, child)
                self_val = self.history[layer][parent][child]

                if self.is_my_turn(layer):
                    cum_val = self_val - expected_val
                    if cum_val > best_value:
                        best_value = cum_val
                        best_move = next_move
                else:
                    cum_val = self_val - expected_val
                    if cum_val > best_value:
                        best_value = cum_val
                        best_move = next_move

            return best_value, best_move

    def distance_to_move(self, move_a, move_b):
        if move_a and move_b:
            arow, acol = move_a
            brow, bcol = move_b
            return abs((arow - brow)) + abs((acol - bcol))
        else:
            return 0

    def get_max_and_closest_move(self, dictionary, prev_move):
        if prev_move == (-1, -1):
            prev_move = self.root_environment.prev_opponents_move
        max_value = max(dictionary.items(), key=lambda x: x[1])[1]
        distances = {}
        for entry in dictionary.keys():
            if dictionary[entry] == max_value:
                distances[entry] = self.distance_to_move(prev_move, entry)
        min_distance_move = min(distances.items(), key=lambda x: x[1])[0]
        return min_distance_move

    def get_move_for_greedy_opponent(self, layer, parent, prev_move=False):
        if parent == ANYMOVE:
            return PASSSCORE, ANYMOVE
        if layer == self.max_depth:
            if prev_move:
                history_copy = copy.deepcopy(self.history[layer][parent])
                if prev_move in history_copy:
                    history_copy[prev_move] = -np.inf
                # best_move = max(history_copy.items(),key=operator.itemgetter(1))[0]
                best_move = self.get_max_and_closest_move(history_copy, parent)
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move
            else:
                # best_move = max(self.history[layer][parent].items(),key=operator.itemgetter(1))[0]
                history_copy = copy.deepcopy(self.history[layer][parent])
                best_move = self.get_max_and_closest_move(history_copy, parent)
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move
        else:
            best_value = -np.inf
            best_cum_value = -np.inf
            best_move = None
            history = copy.deepcopy(self.history)

            if prev_move == False:
                pass
            else:
                if prev_move in history[layer][parent]:
                    history[layer][parent][prev_move] = -np.inf  # mb just remove is safer

            # стратегия жадного игрока
            if not self.is_my_turn(layer):
                best_move = max(self.history[layer][parent].items(), key=operator.itemgetter(1))[
                    0]  # operation on empty set
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move

            expected_opponent_value, expected_opponent_move = 0, 0

            for child in history[layer][parent]:
                # -----------PREVENTING CONSIDERATION OF THE SAME MOVE TWICE
                if layer == 0:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child)  # here is max() on empty set
                else:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child, parent)
                # ----------------------------------------------------------
                my_future_value, my_future_move = 0, 0
                futute_layer = layer + 2
                if self.max_depth >= futute_layer:
                    my_future_value, my_future_move = self.get_move_for_greedy_opponent(futute_layer,
                                                                                        expected_opponent_move, child)
                else:
                    my_future_value = 0
                    my_future_move = False
                # _______________________________

                self_value = history[layer][parent][child]

                if my_future_move == False:
                    cum_value = self_value - expected_opponent_value
                else:
                    cum_value = self_value - expected_opponent_value + my_future_value
                if cum_value > best_cum_value:
                    best_cum_value = cum_value
                    best_value = self_value
                    best_move = child

                    # Тут возможны дву стратегии:
                    #  1) Тупой противник, выбирающий наилучший ход с точки зрения моментальной выгоды
                    #  2) Тот, что скорее всего будет в минимаксе - который вибирает наихудший для меня
                    # сначала поробую реализовать тупого, потом нормальный минимакс
                    # тупой не смотрит на разницу поэтому тут пусто, но для нормального соперника надо смотреть местную разницу

            return best_cum_value, best_move

    def avg_score(self, dict, prev_move=False):
        sum = 0
        number_of_moves = 0
        for move in dict:
            if prev_move:
                if move == prev_move:
                    continue
            sum += dict[move]
            number_of_moves += 1
        return sum / number_of_moves

    def get_move_for_greedy_avg(self, layer, parent, prev_move=False):
        if parent == ANYMOVE:
            return PASSSCORE, ANYMOVE
        if layer == self.max_depth:
            if prev_move:
                history_copy = copy.deepcopy(self.history[layer][parent])
                if prev_move in history_copy:
                    history_copy[prev_move] = -np.inf
                # best_move = max(history_copy.items(),key=operator.itemgetter(1))[0]
                best_move = self.get_max_and_closest_move(history_copy, parent)
                best_value = self.avg_score(self.history[layer][parent], prev_move)
                return best_value, best_move
            else:
                # best_move = max(self.history[layer][parent].items(),key=operator.itemgetter(1))[0]
                history_copy = copy.deepcopy(self.history[layer][parent])
                best_move = self.get_max_and_closest_move(history_copy, parent)
                best_value = self.avg_score(self.history[layer][parent], prev_move)
                return best_value, best_move
        else:
            best_value = -np.inf
            best_cum_value = -np.inf
            best_move = None
            history = copy.deepcopy(self.history)

            if prev_move == False:
                pass
            else:
                if prev_move in history[layer][parent]:
                    history[layer][parent][prev_move] = -np.inf  # mb just remove is safer

            # стратегия жадного игрока
            if not self.is_my_turn(layer):
                best_move = max(self.history[layer][parent].items(), key=operator.itemgetter(1))[
                    0]  # operation on empty set
                best_value = self.avg_score(self.history[layer][parent], prev_move)
                return best_value, best_move

            expected_opponent_value, expected_opponent_move = 0, 0

            for child in history[layer][parent]:
                # -----------PREVENTING CONSIDERATION OF THE SAME MOVE TWICE
                if layer == 0:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child)  # here is max() on empty set
                else:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child, parent)
                # ----------------------------------------------------------
                my_future_value, my_future_move = 0, 0
                futute_layer = layer + 2
                if self.max_depth >= futute_layer:
                    my_future_value, my_future_move = self.get_move_for_greedy_opponent(futute_layer,
                                                                                        expected_opponent_move, child)
                else:
                    my_future_value = 0
                    my_future_move = False
                # _______________________________

                self_value = history[layer][parent][child]

                if my_future_move == False:
                    cum_value = self_value - expected_opponent_value
                else:
                    cum_value = self_value - expected_opponent_value + my_future_value
                if cum_value > best_cum_value:
                    best_cum_value = cum_value
                    best_value = self_value
                    best_move = child

                    # Тут возможны дву стратегии:
                    #  1) Тупой противник, выбирающий наилучший ход с точки зрения моментальной выгоды
                    #  2) Тот, что скорее всего будет в минимаксе - который вибирает наихудший для меня
                    # сначала поробую реализовать тупого, потом нормальный минимакс
                    # тупой не смотрит на разницу поэтому тут пусто, но для нормального соперника надо смотреть местную разницу

            return best_cum_value, best_move


if __name__ == "__main__":
    start_time = time.time()
    environment = Game(BOARD_SIZE)
    environment.read_states()
    depth = environment.set_depth()

    if environment.n_move in [0, 1, 2, 3, 4, 5]:
        actor = NegMaxAgent(1, environment)
        mvs = environment.get_available_moves()
        score, move = actor.get_move()
        layers_to_delete = []
        aquaried_depth = -1
        current_winner = environment.get_winner_label()
        score_diff = environment.get_score_difference()

        for layer in actor.history:
            if len(actor.history[layer]) > 0:
                aquaried_depth += 1
            elif len(actor.history[layer]) == 0:
                layers_to_delete.append(layer)

        actor.max_depth = aquaried_depth

        for layer in layers_to_delete:
            del actor.history[layer]

        # score, move = actor.get_move_for_greedy_opponent(0, (-1, -1), environment.prev_move)
        score, move = actor.get_move_for_greedy_avg(0, (-1, -1), environment.prev_move)

        with open('output.txt', 'w') as output:
            row, col = move
            output.write(str(row) + ',' + str(col))
            environment.commit_move_info(row, col)

        with open('move_time', 'a') as move_info:
            move_info.write(str(time.time() - start_time) + "\n")
    else:

        actor = NegMaxAgent(environment.set_depth(), environment)
        mvs = environment.get_available_moves()
        score, move = actor.get_move()  # create history dict
        layers_to_delete = []
        aquaried_depth = -1
        current_winner = environment.get_winner_label()
        score_diff = environment.get_score_difference()

        for layer in actor.history:
            if len(actor.history[layer]) > 0:
                aquaried_depth += 1
            elif len(actor.history[layer]) == 0:
                layers_to_delete.append(layer)

        actor.max_depth = aquaried_depth

        for layer in layers_to_delete:
            del actor.history[layer]

        # score, move = actor.get_move_for_greedy_opponent(0, (-1, -1), environment.prev_move)
        score, move = actor.get_move_for_greedy_avg(0, (-1, -1), environment.prev_move)

        is_pass_condition = False

        if environment.are_states_identical(environment.current_state, environment.prev_state) and \
                environment.get_winner_label() == environment.player_label:
            is_pass_condition = True
            move = environment.prev_move
        elif aquaried_depth >= 3 and score <= 0.5:
            if environment.get_winner_label() == environment.player_label:
                is_pass_condition = True
                move = environment.prev_move
        elif score < 0 and environment.get_winner_label() == environment.player_label:
            if environment.player_label == 1:
                if environment.get_score_difference() <= 0.5:
                    is_pass_condition = True
                    move = environment.prev_move
            if environment.player_label == 2:
                if environment.get_score_difference() > -0.5:
                    is_pass_condition = True
                    move = environment.prev_move
        elif score < -3:
            if environment.player_label == 1:
                if environment.get_score_difference() <= -0.5:
                    is_pass_condition = True
                    move = environment.prev_move
        elif environment.player_label == 1:
            if environment.get_winner_label() == environment.player_label:
                if score_diff < -0.5:
                    is_pass_condition = True
                    move = environment.prev_move

        with open('output.txt', 'w') as output:
            row, col = move
            if is_pass_condition:
                output.write("PASS")
            else:
                output.write(str(row) + ',' + str(col))
            environment.commit_move_info(row, col)

        with open('move_time', 'a') as move_info:
            move_info.write(str(time.time() - start_time) + "\n")
