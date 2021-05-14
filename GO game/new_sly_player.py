import copy
import os
import random
import numpy as np
import time
import operator

# region
BOARD_SIZE = 5
DEPTH = 2
COMBINATION_RESTRICTION = 11
# ___________________GREADY HEURISTIC__________________________________________________
INITVALUE = 0.5
ITWOLIBERTIES = 0.48
ITHREELIBERTIES = 0.49

BORDERVALUE = 0.035  # changed from 0.075 to 0.035
QUADRANTVALUE = 0.05

HASNEIGBOR = 0.075  # changed to see how it impacts (was 0.2)

GROUPVALUE = 0.2  # chenged from 0.2 to 0.085 then 0.185
RINGVALUE = 0.215  #
CIRCLEVALUE = 0.25  # TODO: NEW
DIAMONDVALUE = 0.2
HALFDIAMONDVALUE = 0.115
SIDEBREAKER = 0.085
SINGLEENEMYVALUE = 0.01

WIPE_MAGNITUDE = 0.3
SAVEMAGNITUDE = 0.2
CIRCLEMAGNITUDE = 0.05  # TODO NEW
SCOREBASE = 0.3

WINNER_SCORE = 100
LOSER_SCORE = -100
PASSSCORE = -20
# ___________________QLEARNER_____________________________________________________________
WREWARD = 1
DREWARD = 0.5
LREWARD = -0.1
ALPHA = 0.7
GAMMA = 0.8
EPSILON = 0.0  # RANDOMNESS IS ALMOST OFF

IVAL = 0.5

ANYMOVE = (101, 101)


# endregion
class Game:
    def __init__(self, N):
        self.N = N
        self.prev_move_file = 'prev_move2.txt'
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
        self.depth_limit = 0

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

    def liberty_second_phase(self, i, j, opponent_label):
        board = self.current_copy()
        board[i][j] = self.player_label
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
        return board, len(chips_to_wipe)

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
        chips_to_wipe = self.chips_to_wipe(self.opponent_label, board)
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

    def get_best_evaluated_move(self, environment, moveset, current_depth, parent_move, last_move=False):
        if moveset == []:
            self.history[current_depth][parent_move][ANYMOVE] = PASSSCORE
            return PASSSCORE, ANYMOVE
        if last_move == False:
            best_value = -np.inf
            move = None

            for row, col in moveset:
                nbs = environment.get_all_neighbors(row, col)
                dgs = environment.get_diagonal_neigbors(row, col)
                # TODO: changed
                move_eval = self.border_init(row, col, environment, nbs)
                # move_eval += self.quadrants_init(row, col, environment)             #TODO % having Q agent i dont think it is sound to have this evaluation anymore
                move_eval += self.group_move_init(row, col, environment)
                move_eval += self.cut_init(row, col, environment, nbs, dgs)  # TODO: nbs and dgs now are parameters

                if environment.player_label == 2:
                    move_eval *= 1.1

                move_eval += self.save_score_init(row, col, environment)
                move_eval += self.get_score_init(row, col, environment)
                move_eval += self.has_neigbor(row, col, environment, nbs)

                # TODO: this is to test should also get circle
                move_eval += self.breake_circle_init(row, col, environment, current_depth)
                move_eval += self.acquire_circle_init(row, col, environment, current_depth)
                # _____________________
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
                        self.history[current_depth][parent_move][(row, col)] = LOSER_SCORE
                return best_value, best_move
            else:
                return 0, "PASS"

    # DFS if can find two paths to border form (row,col) - that means that a circle
    # TODO% need to know square under the circle
    def breake_circle_init(self, row, col, env, current_depth):
        finite_states = []
        adjacent_enemies = env.get_adjacent_allies(row, col, env.opponent_label, env.current_state)
        adjacent_dgn_enemies = env.get_adjacent_diagonal_allies(row, col, env.opponent_label, env.current_state)
        all_adjacent_enemies = adjacent_enemies + adjacent_dgn_enemies

        frontier = adjacent_enemies + adjacent_dgn_enemies + [(row, col)]
        visited = []

        side_A = None
        side_B = None

        while frontier:
            chip = frontier.pop()
            visited.append(chip)
            if chip != (row, col):  # the only not opponent chip in the set
                erow, ecol = chip
                adjacent_enemy_allies = env.get_adjacent_allies(erow, ecol, env.opponent_label, env.current_state)
                adjacent_dgn_enemies = env.get_adjacent_diagonal_allies(erow, ecol, env.opponent_label,
                                                                        env.current_state)
                all_adjacent_enemies = adjacent_enemy_allies + adjacent_dgn_enemies

            if env.is_cell_on_border(chip) and chip not in finite_states:
                if len(finite_states) == 0:
                    finite_states.append(chip)
                    side_A = set(visited)
                elif len(finite_states) == 1:
                    if env.distance_to_move(chip[0], chip[1], finite_states[0]) > 0:
                        finite_states.append(chip)
                        side_B = set(visited)

                if chip not in visited:
                    visited.append(chip)
                if len(finite_states) >= 2:
                    altboard = env.current_copy()
                    path = side_A.union(side_B)
                    altboard[row][col] = env.opponent_label
                    area = self.find_circled_area(finite_states[0], finite_states[1], path)

                    if len(area) > 0:
                        return CIRCLEVALUE
                    else:
                        return 0

            for enemy in all_adjacent_enemies:
                if enemy not in frontier and enemy not in visited:
                    frontier.append(enemy)
        return 0

    # TEST______ACQUAIRE CIRCLE
    def acquire_circle_init(self, row, col, env, current_depth):
        finite_states = []
        adjacent_allies = env.get_adjacent_allies(row, col, env.player_label, env.current_state)
        adjacent_dgn_allies = env.get_adjacent_diagonal_allies(row, col, env.player_label, env.current_state)
        all_adjacent_allies = adjacent_allies + adjacent_dgn_allies

        frontier = adjacent_allies + adjacent_dgn_allies + [(row, col)]
        visited = []

        side_A = None
        side_B = None

        while frontier:
            chip = frontier.pop()
            visited.append(chip)
            if chip != (row, col):  # if not consider it then those on border wont be counted!!!
                erow, ecol = chip
                adjacent_my_allies = env.get_adjacent_allies(erow, ecol, env.player_label, env.current_state)
                adjacent_dgn_allies = env.get_adjacent_diagonal_allies(erow, ecol, env.player_label, env.current_state)
                all_adjacent_allies = adjacent_my_allies + adjacent_dgn_allies

            if env.is_cell_on_border(chip) and chip not in finite_states:
                if len(finite_states) == 0:
                    finite_states.append(chip)
                    side_A = set(visited)
                elif len(finite_states) == 1:
                    if env.distance_to_move(chip[0], chip[1], finite_states[0]) > 0:
                        finite_states.append(chip)
                        side_B = set(visited)

                if chip not in visited:
                    visited.append(chip)
                if len(finite_states) >= 2:
                    altboard = env.current_copy()
                    path = side_A.union(side_B)
                    altboard[row][col] = env.opponent_label
                    area = self.find_circled_area(finite_states[0], finite_states[1], path)

                    if len(area) > 0:
                        return CIRCLEVALUE
                    else:
                        return 0

            for ally in all_adjacent_allies:
                if ally not in frontier and ally not in visited:
                    frontier.append(ally)
        return 0

    def is_prev_circle_subset(self, area, current_depth):
        previous_depth = current_depth - 1
        area_set = set(area)

        for existing_circle_list in self.circles[previous_depth]:
            existing_area_set = set(existing_circle_list)
            if area_set.issubset(existing_circle_list):
                return True
        return False

    def find_circled_area(self, vertex_a, vertex_b, path):
        area = set()
        if vertex_a[0] == vertex_b[0] == 0:
            # upper bound
            area = self.upper_bound_area(path)
        elif vertex_a[0] == vertex_b[0] == (BOARD_SIZE - 1):
            # bottom bound
            area = self.bottom_bound_area(path)
        else:
            # either bottom or upper chose is - min since take over smaller region is easier
            right_bound_area = self.right_bound_area(path)
            left_bound_area = self.left_bound_area(path)

            if len(right_bound_area) < len(left_bound_area):
                area = right_bound_area
            else:
                area = left_bound_area
        return area

    def upper_bound_area(self, path):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            upper_row = vrow - 1
            while upper_row >= 0:
                if (upper_row, vcol) not in path:
                    area.add((upper_row, vcol))
                upper_row -= 1
        return area

    def bottom_bound_area(self, path):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            down_row = vrow + 1
            while down_row <= (BOARD_SIZE - 1):
                if (down_row, vcol) not in path:
                    area.add((down_row, vcol))
                down_row += 1
        return area

    def right_bound_area(self, path):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            right_col = vcol + 1
            while right_col <= (BOARD_SIZE - 1):
                if (vrow, right_col) not in path:
                    area.add((vrow, right_col))
                right_col += 1
        return area

    def left_bound_area(self, path):
        area = set()
        for vertex in path:
            vrow, vcol = vertex
            left_col = vcol - 1
            while left_col >= 0:
                if (vrow, left_col) not in path:
                    area.add((vrow, left_col))
                left_col -= 1
        return area

    # NBS - parameter
    def has_neigbor(self, row, col, env, nbs):
        my_chips = 0
        for i in range(env.N):
            for j in range(env.N):
                if env.current_state[i][j] == env.player_label:
                    my_chips += 1

        if my_chips == 1:
            return HASNEIGBOR / 2
        elif my_chips == 2:
            return HASNEIGBOR
        return 0

    def quadrants_init(self, move_row, move_col, env):
        quadrants = [(1, 1), (1, 3), (3, 1), (3, 3)]

        if (move_row, move_col) in quadrants:
            neigbors = env.get_all_neighbors(move_row, move_col)
            diag_neigbors = env.get_diagonal_neigbors(move_row, move_col)

            if len(neigbors) + len(diag_neigbors) == 8:
                return QUADRANTVALUE
        return 0

    def border_init(self, row, col, env, nbs):
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
            return GROUPVALUE
        elif distance == 2:
            if len(env.get_all_neighbors(row, col)) < 4:
                return self.distance_fitness(distance) + BORDERVALUE
            return self.distance_fitness(distance)
        return self.distance_fitness(distance)

    # TODO:!!!! CHENCGED nbs and dgs are added as a parameter
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

        if enemy_diag_chips == 3 and neutral_diag_chips == 3 and neutral_chips == 3:
            return RINGVALUE
        elif enemy_diag_chips == 3 and my_diag_chips == 1:
            if neutral_chips == 4 or neutral_chips == 3:
                return RINGVALUE
            else:
                return RINGVALUE / 4
        # TODO NEW
        elif enemy_diag_chips == 2 and neutral_chips == 3 and (
                neutral_diag_chips + my_diag_chips + enemy_diag_chips) == 2:
            return RINGVALUE
        elif my_diag_chips == 2 and neutral_chips == 3 and (neutral_diag_chips + my_diag_chips + enemy_diag_chips) == 2:
            return RINGVALUE
        # TODO NEW
        # TODO: >= 0 my?
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
        # elif len(nbs) == 3 and enemy_chips == 1 and my_diag_chips == 1:
        #    return RINGVALUE   #or diamond?
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

    def save_score_init(self, row, col, env):
        loss = env.is_potential_loss(row, col)
        if loss:
            if env.is_move_valid(row, col, opponent_check=True):
                return SCOREBASE + loss * SAVEMAGNITUDE
        return 0

    def get_score_init(self, row, col, env):
        score = env.is_potential_score(row, col)
        if score:
            return SCOREBASE + score * WIPE_MAGNITUDE
        return 0

    def is_my_turn(self, depth):
        if depth % 2 == 0: return True
        return False

    # TODO tuned
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

    # TODO: KO RULE!
    def get_move_for_greedy_opponent(self, layer, parent, prev_move=False):
        if parent == ANYMOVE:
            return PASSSCORE, ANYMOVE
        if layer == self.max_depth:
            if prev_move:
                history_copy = copy.deepcopy(self.history[layer][parent])
                if prev_move in history_copy:
                    history_copy[prev_move] = -np.inf
                best_move = max(history_copy.items(), key=operator.itemgetter(1))[0]  # operation on empty set
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move
            else:
                best_move = max(self.history[layer][parent].items(), key=operator.itemgetter(1))[
                    0]  # operation on empty set
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move
        else:
            best_value = -np.inf
            best_cum_value = -np.inf
            best_move = None

            # стратегия жадного игрока
            if not self.is_my_turn(layer):
                best_move = max(self.history[layer][parent].items(), key=operator.itemgetter(1))[
                    0]  # operation on empty set
                best_value = self.history[layer][parent][best_move]
                return best_value, best_move

            expected_opponent_value, expected_opponent_move = 0, 0
            history = copy.deepcopy(self.history)

            if prev_move == False:
                pass
            else:
                if prev_move in history[layer][parent]:
                    history[layer][parent][prev_move] = -np.inf  # mb just remove is safer

            for child in history[layer][parent]:
                # -----------PREVENTING CONSIDERATION OF THE SAME MOVE TWICE
                if layer == 0:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child)  # here is max() on empty set
                else:
                    expected_opponent_value, expected_opponent_move = self.get_move_for_greedy_opponent(layer + 1,
                                                                                                        child,
                                                                                                        parent)
                # ----------------------------------------------------------

                # TEST учет следующего моего хода
                my_future_value, my_future_move = 0, 0
                futute_layer = layer + 2
                if self.max_depth >= futute_layer:
                    my_future_value, my_future_move = self.get_move_for_greedy_opponent(futute_layer,
                                                                                        expected_opponent_move, child)
                else:
                    my_future_value = 0
                    my_future_move = False
                # _______________________________

                self_value = self.history[layer][parent][child]
                check = 1
                if my_future_move == False:
                    cum_value = self_value - expected_opponent_value
                else:
                    cum_value = self_value - expected_opponent_value + my_future_value
                if cum_value > best_cum_value:
                    best_cum_value = cum_value
                    best_value = self_value
                    best_move = child  # IMPORTANT DIFFERENCE WHICH I MAY HAVE MISSED IN PREV VERSION OF MINIMAX
                    check = 1
                    # Тут возможны дву стратегии:
                    #  1) Тупой противник, выбирающий наилучший ход с точки зрения моментальной выгоды
                    #  2) Тот, что скорее всего будет в минимаксе - который вибирает наихудший для меня
                    # сначала поробую реализовать тупого, потом нормальный минимакс
                    # тупой не смотрит на разницу поэтому тут пусто, но для нормального соперника надо смотреть местную разницу

            return best_cum_value, best_move

    def check(self, layer, parent):
        if layer == self.max_depth:
            hist = self.history
            best_move = max(self.history[layer][parent])
            check = 1
        else:
            for child in self.history[layer][parent]:
                self.check(layer + 1, child)


class QAgent:
    def __init__(self, qtable='qtable.txt', alpha=ALPHA, gamma=GAMMA, init_val=IVAL, label=None, env=None,
                 load_table=True):
        self.label = label
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history = []  # TODO: history should be written to file line by line
        self.init_val = init_val  # TODO: add initialization from the file method
        self.qtable_file = qtable
        self.history_file = 'game_history2.txt'
        self.table_size = 0
        if load_table:
            self.load_qtable()
            self.table_size = len(self.q_values)
        self.environment = env

    def get_Q_values(self, state):
        if not isinstance(state, str):
            state = self.state_to_string(state)

        if state not in self.q_values:
            q_val = np.zeros((BOARD_SIZE, BOARD_SIZE))
            q_val.fill(self.init_val)
            self.q_values[state] = q_val

            with open(self.qtable_file, 'a') as qtable:
                qtable.write(state + " " + self.q_val_to_string(q_val))
                qtable.write('\n')

        return self.q_values[state]

    def select_move(self, moveset):
        state = self.environment.current_state
        q_values = self.get_Q_values(state)
        current_max = -np.inf
        move_row, move_col = 0, 0
        same_values = []

        for row, col in moveset:
            if q_values[row][col] > current_max:
                current_max = q_values[row][col]
                move_row, move_col = row, col

                same_values = []
                same_values.append((row, col))
            elif q_values[row][col] == current_max:
                same_values.append((row, col))

        if len(same_values) > 1:
            move_row, move_col = random.choice(same_values)

        return move_row, move_col

    def move(self, moveset):
        chance = random.random()
        if moveset == []:
            return "PASS"
        else:
            chance = random.random()
            if chance < EPSILON:
                row, col = random.choice(moveset)
            else:
                row, col = self.select_move(moveset)
            with open(self.history_file, 'a') as hfile:
                hfile.write(
                    self.state_to_string(self.environment.current_state) + " " + str(row) + "," + str(col) + " " + str(
                        environment.n_move))
                hfile.write("\n")
            return row, col

    def learn(self, reward):
        self.load_history()
        self.history.reverse()
        max_q = -1.0

        for action in self.history:
            state, move = action
            qvals = self.get_Q_values(state)

            qvals[move[0]][move[1]] = qvals[move[0]][move[1]] * (1 - self.alpha) + \
                                      self.alpha * self.gamma * max_q
            max_q = np.max(qvals)

        os.remove(self.history_file)
        self.update_qtable()

    def get_opponents_reward(self, my_reward):
        if my_reward == LREWARD:
            return WREWARD
        elif my_reward == WREWARD:
            return LREWARD
        else:
            return DREWARD

    def QLearn(self, reward, play_against_myself=False):
        self.load_history()
        self.history.reverse()

        max_q = -1
        iter = 0

        for action in self.history:
            players_state = int(action[0][0])
            if players_state == environment.player_label:
                r = reward
            else:
                r = self.get_opponents_reward(reward)

            state, move, move_number = action

            learning_influencer = self.gamma ** ((BOARD_SIZE * BOARD_SIZE - 1) / 2 - move_number) * r

            qvals = self.get_Q_values(state)

            qvals[move[0]][move[1]] = (1 - self.alpha) * qvals[move[0]][move[1]] + \
                                      self.alpha * (learning_influencer + self.gamma * max_q)
            iter += 1
            max_q = np.max(qvals)
        os.remove(self.history_file)
        self.update_qtable()

    def update_qtable(self, file_name=""):
        if file_name == "":
            file_name = self.qtable_file

        with open(file_name, 'w') as qtable:
            for state in self.q_values:
                qtable.write(state + " " + self.q_val_to_string(self.q_values[state]))
                qtable.write("\n")

    def load_qtable(self):
        if os.path.isfile(self.qtable_file):
            with open(self.qtable_file, 'r') as qtable:
                line = qtable.readline()
                while line:
                    state, qval = line.split(" ")
                    self.q_values[state] = self.string_to_qval(qval)
                    line = qtable.readline()

    def load_history(self):
        with open(self.history_file, 'r') as hist:
            line = hist.readline()

            while line:
                state, move, move_number = line.split(" ")
                row, col = move.split(',')
                row, col = int(row), int(col)
                move_number = int(move_number)
                self.history.append((state, (row, col), move_number))

                line = hist.readline()

    def state_to_string(self, state):
        return str(environment.player_label) + ''.join([str(state[i][j]) for i in range(BOARD_SIZE)
                                                        for j in range(BOARD_SIZE)]
                                                       )

    def q_val_to_string(self, q_values):
        return ''.join([str(q_values[i][j]) + '|' for i in range(BOARD_SIZE)
                        for j in range(BOARD_SIZE)]
                       )

    def string_to_qval(self, string):
        qvals = string.split('|')
        result = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                result[i][j] = qvals[BOARD_SIZE * i + j]
        return result


def ManualInput():
    key = input("Key:")
    environment = Game(BOARD_SIZE)
    environment.read_states()
    actor = NegMaxAgent(environment.set_depth(), environment)
    cs = environment.current_state

    while key != 'q:':
        if key == 'tree':
            score, move = actor.get_move()
            layers_to_delete = []
            aquaried_depth = -1

            for layer in actor.history:
                if len(actor.history[layer]) > 0:
                    aquaried_depth += 1
                elif len(actor.history[layer]) == 0:
                    layers_to_delete.append(layer)

            actor.max_depth = aquaried_depth

            for layer in layers_to_delete:
                del actor.history[layer]

            score, move = actor.get_move_for_greedy_opponent(0, (-1, -1))
            cjef = 1
            key = input("Key:")
        else:
            row, col, player = key

            row = int(row)
            col = int(col)
            player = int(player)

            cs[row][col] = player
            if player == 1:
                cwp = environment.chips_to_wipe(2, cs)
            else:
                cwp = environment.chips_to_wipe(1, cs)

            environment.move_number_incrementation()
            environment.wipe(cwp, cs)
            asfas = 1
            key = input("Key:")


# 25046  next line  25047

if __name__ == "__main__":
    start_time = time.time()
    environment = Game(BOARD_SIZE)
    environment.read_states()

    if environment.n_move in [0, 1, 2, 3, 4, 5]:
        moveset = environment.get_available_moves()

        actor = QAgent(label=environment.player_label, env=environment, qtable='qtable_inverted_addition.txt')
        move = actor.move(moveset)

        if move == "PASS":
            with open('output.txt', 'w') as output:
                output.write("PASS")
        else:
            row, col = move
            with open('output.txt', 'w') as output:
                output.write(str(row) + "," + str(col))

            environment.commit_move_info(row, col)
    else:
        actor = NegMaxAgent(environment.set_depth(),
                            environment)  # TODO: Loop was in set_depth() проверить как на маке!!!! ПОчему такая проблема не возникалка на вокариуме?
        actor.forbidden_move = environment.prev_move
        mvs = environment.get_available_moves()
        score, move = actor.get_move()  # create history dict
        layers_to_delete = []
        aquaried_depth = -1

        for layer in actor.history:
            if len(actor.history[layer]) > 0:
                aquaried_depth += 1
            elif len(actor.history[layer]) == 0:
                layers_to_delete.append(layer)

        actor.max_depth = aquaried_depth

        for layer in layers_to_delete:
            del actor.history[layer]

        score, move = actor.get_move_for_greedy_opponent(0, (-1, -1))

        with open('output.txt', 'w') as output:
            row, col = move
            output.write(str(row) + ',' + str(col))
            environment.commit_move_info(row, col)

    with open('move_time', 'a') as move_info:
        move_info.write(str(time.time() - start_time) + "\n")
