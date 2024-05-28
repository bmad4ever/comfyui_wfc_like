from collections import defaultdict
from functools import lru_cache, cache
from multiprocessing.shared_memory import ShareableList
import numpy as np
from numpy import ndarray
from py_search.base import Problem, Node
import hashlib

TILE_DIGEST_SIZE = 4  # in bytes
NP_ENCODED_TILE_TYPE = "longlong"


class WFC_Sample:
    """
    From a source image, compute & store the following data:
        tile_data : { tile hashcode : ( tile , frequency )  , ... }
        super_tile_data : [ ( 3x3 matrix of tile hashes, count ) ]
        tile_dims : ( tile height, width, channels )
    """

    @staticmethod
    def tile_to_hash(tile):
        return int.from_bytes(hashlib.blake2b(tile.tobytes(), digest_size=TILE_DIGEST_SIZE).digest(), byteorder="big")

    def __init__(self, src_imgs, cell_width, cell_height):
        self.tile_data, self.super_tile_data, self.tile_dims = self.prepare(src_imgs[0], cell_width, cell_height)

        for img in src_imgs[1:]:
            r_tile_data, r_super_tile_data, _ = self.prepare(img, cell_width, cell_height)
            self.tile_data = {k: (v[0], self.tile_data.get(k, (None, 0))[1] + v[1])
                              for k, v in {**self.tile_data, **r_tile_data}.items()}
            self.super_tile_data = self.merge_tuples(self.super_tile_data, r_super_tile_data)

    @staticmethod
    def merge_tuples(list1, list2):  # adapted from GPT; might be wrong
        result_dict = defaultdict(int)
        shape = list1[0][0].shape
        for ndarray, value in list1 + list2:
            # Convert ndarray to a hashable type (tuple)
            key = tuple(ndarray.flatten())
            result_dict[key] += value

        # Convert the result back to a list of tuples
        result_list = [(np.array(key).reshape(shape), value) for key, value in result_dict.items()]
        return result_list

    def get_tile_data(self) -> dict[int, tuple[ndarray, float]]:
        """
        @return: hash to (tile, freq) pair dictionary
        """
        return self.tile_data

    def get_super_tile_data(self) -> list[tuple[ndarray, int]]:
        """
        @return: list of (super tile, count) pairs
        """
        return self.super_tile_data

    @staticmethod
    def adjust_image_to_tile_size(img, tile_height, tile_width):
        """
        @return: the adjusted image, height in number of cells, and width
        """
        if img.shape[0] % tile_height != 0:
            print(f"src height ({img.shape[0]}) is not divisible by cell_height ({tile_height})!")
        if img.shape[1] % tile_width != 0:
            print(f"src width ({img.shape[1]}) is not divisible by cell_height ({tile_width})!")

        height_in_tiles = img.shape[0] // tile_height
        width_in_tiles = img.shape[1] // tile_width
        assert height_in_tiles >= 3 and width_in_tiles >= 3, "sample too small to infer adjacency rules."

        new_height = tile_height * height_in_tiles
        new_width  = tile_width * width_in_tiles
        adjusted_image = img.copy()[0:new_height, 0:new_width, :]

        return adjusted_image, height_in_tiles, width_in_tiles

    @staticmethod
    def prepare(src_img, tile_width, tile_height):
        src_shape = src_img.shape
        adjusted_img, ycell_len, xcell_len = WFC_Sample.adjust_image_to_tile_size(src_img, tile_height, tile_width)
        tiles = adjusted_img.reshape(
            (
                ycell_len,  # adjusted_img.shape[0] // tile_height,
                tile_height,
                xcell_len,  # adjusted_img.shape[1] // tile_width,
                tile_width,
                src_shape[2]
            )
        ).swapaxes(1, 2)
        size_in_tiles = tiles.shape[:2]
        tiles = tiles.reshape(-1, tile_height, tile_width, src_shape[2])
        utiles, counts = np.unique(tiles, axis=0, return_counts=True)
        ut_hashes = [WFC_Sample.tile_to_hash(tile) for tile in utiles]
        # assert len(set(ut_hashes)) == len(ut_hashes)

        tiles_data = dict(zip(ut_hashes, zip(utiles, counts / tiles.shape[0])))
        hashed_tiles = np.array([WFC_Sample.tile_to_hash(tile) for tile in tiles])
        hashed_tiles = hashed_tiles.reshape(size_in_tiles)

        super_tiles = np.array(
            [hashed_tiles[y:y + 3, x:x + 3] for y in range(size_in_tiles[0] - 2) for x in range(size_in_tiles[1] - 2)])

        u_super_tiles, super_counts = np.unique(super_tiles, axis=0, return_counts=True)
        super_tiles_data = list(zip(u_super_tiles, super_counts))

        return tiles_data, super_tiles_data, (tile_height, tile_width, src_shape[2])

    def img_to_tile_encoded_world(self, src_img):
        adjusted_img, ycell_len, xcell_len = WFC_Sample.adjust_image_to_tile_size(src_img, *self.tile_dims[:2])
        tiles = adjusted_img.reshape(
            (
                ycell_len,
                self.tile_dims[0],
                xcell_len,
                self.tile_dims[1],
                adjusted_img.shape[2]
            )
        ).swapaxes(1, 2)
        size_in_tiles = tiles.shape[:2]
        tiles = tiles.reshape(-1, self.tile_dims[0], self.tile_dims[1], adjusted_img.shape[2])
        hashed_world = [WFC_Sample.tile_to_hash(tile) for tile in tiles]
        hashed_world = np.array([tile if tile in self.tile_data.keys() else 0 for tile in hashed_world]).astype(
            NP_ENCODED_TILE_TYPE)
        return hashed_world.reshape(*size_in_tiles)

    def tile_encoded_to_img(self, src_state: ndarray):
        th, tw = self.tile_dims[:2]
        img = np.zeros((src_state.shape[0] * th, src_state.shape[1] * tw, self.tile_dims[2]))
        mask = np.ones((src_state.shape[0] * th, src_state.shape[1] * tw)) * 255
        for (y, x), hashcode in np.ndenumerate(src_state):
            if hashcode in self.tile_data:
                img[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = self.tile_data[hashcode][0]
                mask[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = np.zeros(self.tile_data[hashcode][0].shape[:2])
        return img, mask


class WFC_Problem(Problem):
    def __init__(self, sample: WFC_Sample, starting_state: ndarray, seed: int = 0, use_8_cardinals: bool = False,
                 relax_validation: bool = False, max_freq_adjust: float = 1, plateau_check_interval: int = -1,
                 starting_temperature: float = 50, min_min_temperature: float = 0, max_min_temperature: float = 80,
                 reverse_depth_w: float = 1, node_cost_w: float = 1, prev_state_avg_entropy_w: float = 0,
                 stop_and_ticker_shm_list: ShareableList = None, pid: int = 0
                 ):
        """
        @param sample: contains the tiles, their frequencies and "implicit" constraints
        @param starting_state: complete the provided state instead of starting with an empty world.
                                If provided, width and height are ignored.
        @param seed: used to set up the generator so that the result can be reproducible ( deterministic )
        @param use_8_cardinals: consider the surrounding 8 tiles if set to TRUE; OR only the 4 cardinals if set to FALSE
        @param max_freq_adjust: scale frequency adjustment weight.
                                if set to zero, the frequency of the tiles registered in the given samples is ignored.
        @param plateau_check_interval: the number of nodes to be processed before checking the highest registered depth.
                                       if the depth hasn't changed between checks, the search is stopped.
                                       Set to 0 (zero) to ignore plateau checks.
                                       Set to -1 (minus 1) to auto select depending on the number of nodes to process.
        @param starting_temperature:
        @param min_min_temperature:
        @param max_min_temperature:
        @param reverse_depth_w:
        @param node_cost_w:
        @param prev_state_avg_entropy_w:

        @param stop_and_ticker_shm_list: shared memory list.
                                        element at index=0 indicates whether to execution as been canceled or not.
                                        elements at index>1 will store the best depth for each of the generations.
        """
        super().__init__(initial=(0, 0), initial_cost=0, extra=0)
        # initial state -> (depth, hash) -> 0 represents empty world at the start, at zero depth
        # extra -> the sum of entropies of the closed nodes in the current branch ( at the moment they were closed )

        non_zeroes = np.count_nonzero(starting_state)
        self._number_of_tiles_to_process = starting_state.size - non_zeroes
        if self._number_of_tiles_to_process == 0:
            self._stop = True
            return

        # BASIC DATA
        self._stop_and_ticker = stop_and_ticker_shm_list
        self._pid = pid
        self.rng = np.random.default_rng(seed=seed)
        self._sample: WFC_Sample = sample
        self._relaxed_validation = relax_validation

        self._world_tdims = starting_state.shape
        self._temp_world_state = starting_state.copy()
        self._starting_state = starting_state.copy() if non_zeroes > 0 else None
        # _starting_state has 2 internal uses:
        # 1. if None the center tile is set to open, otherwise the state is iterated to find the tiles at the edges
        # 2. initialize state to return instead of reverting last node actions

        # KEEP TRACK OF OPEN TILES ( yet to explore after the last closed node )
        self._temp_world_open_tiles: set[tuple[int, int]] = set([])

        # INFLUENCE COST WEIGHTS & FINAL NODE VALUE
        # influences the nodes' costs. high temperature lowers the influence of random noise and frequency adjustments
        self._min_temperature = starting_temperature
        self._min_min_temperature, self._max_min_temperature = min_min_temperature, max_min_temperature
        self._rev_depth_w, self._node_cost_w, self._ps_avg_ent_w = reverse_depth_w, node_cost_w, prev_state_avg_entropy_w

        # STOP THE SEARCH
        self._best_node = None
        self._prev_best_depth = 0
        self._last_node = None
        self._plateau_check_ticker = 0
        self._plateau_stop_steps = self._world_tdims[0] * self._world_tdims[1] / 2.0 \
            if plateau_check_interval == -1 else plateau_check_interval

        self._stop: bool = False  # stops the search; set to True when all the tiles are filled OR a plateau is reached

        # OTHERS
        self._use_8cardinals = use_8_cardinals
        self.get_cell_potential_states = \
            self.get_cell_potential_states_8cardinals if use_8_cardinals\
            else self.get_cell_potential_states_4cardinals

        tile_data = sample.get_tile_data()
        self._tile_counts = dict(zip(tile_data.keys(), [0] * len(tile_data)))
        if self._starting_state is not None:
            for tile in self._starting_state.flat:
                if tile != 0:
                    self._tile_counts[tile] += 1

        self._max_freq_adjust = max_freq_adjust
        t = self._number_of_tiles_to_process
        a = [[0, 0, 1], [t ** 2, t, 1], [(t / 2) ** 2, t / 2, 1]]
        b = [t, t, 0]
        self._tile_freq_adjustment_poly = np.poly1d(np.linalg.solve(a, b))

    def is_generation_aborted(self) -> bool:
        return self._stop_and_ticker is not None and self._stop_and_ticker[0]

    @lru_cache(maxsize=8)
    def temp_ratio(self, node_depth: int, prior_node_depth: int):
        # TODO -> potentially something to change/customize
        depth_diff = prior_node_depth - node_depth
        depth_ratio = node_depth / self._number_of_tiles_to_process
        ratio = min(.9, (abs(depth_diff) * 3) / np.sqrt(self._number_of_tiles_to_process)) if depth_diff != 0 else \
            depth_ratio ** 2.5 / 80
        return ratio

    def get_new_temperature(self, node_depth: int, prior_node_depth: int):
        limit = self._max_min_temperature if node_depth <= prior_node_depth else self._min_min_temperature
        ratio = self.temp_ratio(node_depth, prior_node_depth)
        return limit * ratio + self._min_temperature * (1 - ratio)

    @lru_cache(maxsize=32)
    def _tile_freq_adjustment_func(self, depth):
        return self._max_freq_adjust * (1 - self._tile_freq_adjustment_poly(depth) / self._number_of_tiles_to_process)

    def adjacent_tiles_coords(self, tile_y: int, tile_x: int, get_diagonals: bool) -> list[tuple[int, int]]:
        """
        @param get_diagonals: also return diagonally adjacent tiles?
        @return: a list of tuple pairs with the coordinates of the tiles adjacent to the input tile
        """
        # pre-calculate bounds
        min_y = max(0, tile_y - 1)
        max_y = min(self._temp_world_state.shape[0] - 1, tile_y + 1)
        min_x = max(0, tile_x - 1)
        max_x = min(self._temp_world_state.shape[1] - 1, tile_x + 1)

        return [(adj_y, adj_x) for adj_y in range(min_y, max_y + 1)
                for adj_x in range(min_x, max_x + 1)
                if (adj_y, adj_x) != (tile_y, tile_x)
                and (get_diagonals or (adj_y == tile_y or adj_x == tile_x))
                ]

    def adjacent_tiles_coords_with_outbounds(self, tile_y: int, tile_x: int, get_diagonals: bool) -> list[tuple[int, int]]:
        """
        @param get_diagonals: also return diagonally adjacent tiles?
        @return: a list of tuple pairs with the coordinates of the tiles adjacent to the input tile
        """
        return [(adj_y, adj_x) if
                0 <= adj_y < self._temp_world_state.shape[0] and
                0 <= adj_x < self._temp_world_state.shape[1]
                else None
                for adj_y in range(tile_y - 1, tile_y + 2)
                for adj_x in range(tile_x - 1, tile_x + 2)
                if (adj_y, adj_x) != (tile_y, tile_x)
                and (get_diagonals or (adj_y == tile_y or adj_x == tile_x))
                ]

    @property
    @cache
    def _3x3_adjacency_kernel(self):
        kernel = np.ones((3, 3)) if self._use_8cardinals else np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape((3, 3))
        kernel[1, 1] = 0
        return kernel

    @staticmethod
    def get_5x5_roi(world_state, wx, wy):
        """
        5x5 matrix that is a window into a subsection of the world_state matrix.
        The window is centered at the (wy, wx) coordinates in the world_state.
        Out of bounds cells are set with zeroes.
        """
        return np.pad(world_state[max(0, wy - 2):min(world_state.shape[0], wy + 3),
                      max(0, wx - 2):min(world_state.shape[1], wx + 3)], (
                          (max(2 - wy, 0), max(wy + 3 - world_state.shape[0], 0)),
                          (max(2 - wx, 0), max(wx + 3 - world_state.shape[1], 0))), constant_values=0)

    def close_node(self, y: int, x: int):
        indices_to_check = self.adjacent_tiles_coords(y, x, self._use_8cardinals)
        self._temp_world_open_tiles.remove((y, x))
        for _y, _x in indices_to_check:
            if self._temp_world_state[_y, _x] == 0:
                self._temp_world_open_tiles.add((_y, _x))

    def reopen_node(self, y: int, x: int):
        indices_to_check = self.adjacent_tiles_coords(y, x, self._use_8cardinals)
        self._temp_world_open_tiles.add((y, x))
        for (_y, _x) in indices_to_check:
            if not self._temp_world_open_tiles.__contains__((_y, _x)):
                continue
            sub_indices_to_check = self.adjacent_tiles_coords(_y, _x, self._use_8cardinals)
            if any(self._temp_world_state[__y, __x] != 0 for (__y, __x) in sub_indices_to_check):
                continue  # -> position should remain open
            # otherwise -> position should be closed
            self._temp_world_open_tiles.remove((_y, _x))

    def reopen_node_v2(self, y: int, x: int):
        """
        NOT USED FOR NOW. Might slightly improve performance.
        """
        from scipy.signal import convolve2d

        self._temp_world_open_tiles.add((y, x))

        # use convolution to check adjacency
        kernel = self._3x3_adjacency_kernel
        window_5x5 = self._temp_world_state[max(0, y - 2):min(y + 3, self._temp_world_state.shape[0]),
                     max(0, x - 2):min(x + 3, self._temp_world_state.shape[1])]
        conv_matrix = convolve2d(window_5x5, kernel, mode='valid')

        for i in range(conv_matrix.shape[0]):
            _y = max(1, y - 1) + i
            for j in range(conv_matrix.shape[1]):
                _x = max(1, x - 1) + j
                if (
                        # ugly, but using np.argwhere to build indices seems slower
                        (_y == y and _x == x)
                        or not self._temp_world_open_tiles.__contains__((_y, _x))
                        or (not self._use_8cardinals and kernel[_y - y + 1, _x - x + 1] == 0)
                        or conv_matrix[i, j] > 0
                ):
                    continue
                self._temp_world_open_tiles.remove((_y, _x))

    @cache
    def tile_remains_valid_4cardinals(self, p2, p4, p5, p6, p8):
        """[p1->tl, ..., p9->br] ; where p5 is the center tile"""
        super_tile_data = self._sample.get_super_tile_data()

        def is_possible(super_tile):
            return not (super_tile[1, 1] != p5
                        or p2 != 0 and super_tile[0, 1] != p2 or p4 != 0 and super_tile[1, 0] != p4
                        or p6 != 0 and super_tile[1, 2] != p6 or p8 != 0 and super_tile[2, 1] != p8
                        )

        return any(is_possible(stile) for stile, _ in super_tile_data)

    @cache
    def tile_remains_valid_8cardinals(self, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        super_tile_data = self._sample.get_super_tile_data()

        def is_possible(super_tile):
            return not (super_tile[1, 1] != p5
                        or p1 != 0 and super_tile[0, 0] != p1 or p2 != 0 and super_tile[0, 1] != p2
                        or p3 != 0 and super_tile[0, 2] != p3 or p4 != 0 and super_tile[1, 0] != p4
                        or p6 != 0 and super_tile[1, 2] != p6 or p7 != 0 and super_tile[2, 0] != p7
                        or p8 != 0 and super_tile[2, 1] != p8 or p9 != 0 and super_tile[2, 2] != p9
                        )

        return any(is_possible(stile) for stile, _ in super_tile_data)

    def validate_adjacent(self, tile_data: dict[int, int], world_state: ndarray,
                          indices_to_check: list[tuple[int, int]], wy: int, wx: int):
        """

        @param tile_data: the potential tile types to open at the given world position (wy, wx);
                dict (key-> tile type, value-> counts)
        @param indices_to_check: indices to check surrounding (wy, wx)
        @param wy: tile whose vicinity is to be validated y coordinate in the world
        @param wx: tile whose vicinity is to be validated x coordinate in the world
        @return:
        """
        if len(tile_data) == 0:
            return {}

        roi_matrix = self.get_5x5_roi(world_state, wx, wy)

        def check_if_all_adjacent_tiles_remain_valid_v2(tile_type):
            roi_matrix[2, 2] = tile_type  # simulate tile placement
            for y, x in indices_to_check:
                if roi_matrix[y - wy + 2, x - wx + 2] == 0:
                    continue

                # get the corresponding values from the 5x5 roi_matrix
                adjacent_states = roi_matrix[y - wy + 2 - 1:y - wy + 2 + 2,
                                  x - wx + 2 - 1:x - wx + 2 + 2].flatten().tolist()

                if self._use_8cardinals:
                    if not self.tile_remains_valid_8cardinals(*adjacent_states):
                        return False
                else:
                    if not self.tile_remains_valid_4cardinals(*[adjacent_states[i] for i in [1, 3, 4, 5, 7]]):
                        return False

            return True

        new_tile_data = {k: c for k, c in tile_data.items() if check_if_all_adjacent_tiles_remain_valid_v2(k)}
        return new_tile_data


    def get_cell_potential_states_and_costs(self, y, x, world_state, depth) -> \
            tuple[list[bytes], ndarray | None, float | None]:
        # get adjacents' (y,x) coordinates. if out of bounds get a None instead.
        adjacent_indices = self.adjacent_tiles_coords_with_outbounds(y, x, self._use_8cardinals)
        # build adjacency setting outbounds as zeroes.
        adjacent_states = [0 if idx is None else world_state[idx[0], idx[1]] for idx in adjacent_indices]
        potential_tiles_data = self.get_cell_potential_states(*adjacent_states)  # must be given in correct order

        # check if adjacent, non-empty tiles, remain valid; if not, remove potential tile
        if not self._relaxed_validation:
            adjacent_indices = [idx for idx in adjacent_indices if idx is not None]  # get rid of out of bounds items
            potential_tiles_data = self.validate_adjacent(potential_tiles_data, world_state,
                                                          adjacent_indices, y, x)

        # using the stored sample counts, compute each tile type probability
        tile_types, probabilities = self.map_to_probabilities(potential_tiles_data) or ([], None)

        if probabilities is None:
            return [], None, None  # nothing to compute, so just return early

        # generate random weights and temperature
        rands = self.rng.random(len(probabilities))
        temp = 1 - self.rng.integers(low=int(self._min_temperature), high=100) / 100.0

        # GET tile type freq in original samples AND in current generation
        tile_data = self._sample.get_tile_data()
        sample_freqs = np.array([tile_data[t][1] for t in tile_types])
        current_counts = np.array([self._tile_counts[t] for t in tile_types])
        current_freqs = current_counts / max(1, depth)

        if np.isclose(self._max_freq_adjust, 0.0, rtol=0.0, atol=1.e-8):
            adjusted_freqs_diff = np.zeros(probabilities.size)
        else:
            depth_adjustment = self._tile_freq_adjustment_func(depth)
            adjusted_freqs_diff = np.sign(sample_freqs - current_freqs) * (
                    1 - np.minimum(sample_freqs, current_freqs) / np.maximum(sample_freqs, current_freqs))
            adjusted_freqs_diff *= depth_adjustment

        # compute entropy and node costs
        if len(probabilities) == 1:
            # collapse -> only one possibility
            entropy = 0.0
            costs = 1 - probabilities
        else:
            entropy = - np.sum(probabilities * np.log2(probabilities))
            normalized_entropy = min(1.0, entropy / np.log2(len(probabilities)))
            costs = (1 - np.clip(probabilities
                                 + adjusted_freqs_diff * temp * normalized_entropy
                                 + (rands * 2 - 1) * temp * normalized_entropy
                                 , 0.0001, 1))

        return tile_types, costs, entropy

    @cache
    def get_cell_potential_states_8cardinals(self, p1, p2, p3, p4, p6, p7, p8, p9):
        """
        the state of adjacent cells; where 0 = unknown
        [[1,2,3],
         [4,c,6],
         [7,8,9]]
        @return: dictionary [ key->tile type, value->counts ]
        """

        def is_possible(super_tile):
            tiles = np.array([[p1, p2, p3], [p4, 0, p6], [p7, p8, p9]])
            for (y, x), tile in np.ndenumerate(tiles):
                # print(f"super={super_tile} ;  index = {(y,x)}")
                if tile == 0:
                    continue
                if super_tile[y, x] != tile:
                    return False
            return True

        # filter all the possible super tiles
        pcs = defaultdict(int)
        for stile, count in self._sample.get_super_tile_data():
            if is_possible(stile):
                pcs[stile[1, 1]] += count

        return pcs

    @cache
    def get_cell_potential_states_4cardinals(self, p2, p4, p6, p8):
        """
        Read get_cell_potential_states_8cardinals documentation.
        This function is similar, but only takes into account 4 cardinals
        """

        def is_possible(super_tile):
            tiles = np.array([[0, p2, 0], [p4, 0, p6], [0, p8, 0]])
            for (y, x), tile in np.ndenumerate(tiles):
                if tile == 0:
                    continue
                if super_tile[y, x] != tile:
                    return False
            return True

        # filter all the possible super tiles
        pcs = defaultdict(int)
        for stile, count in self._sample.get_super_tile_data():
            if is_possible(stile):
                pcs[stile[1, 1]] += count

        return pcs

    @staticmethod
    def map_to_probabilities(pcs: dict[int, int]) -> tuple[list[int], ndarray] | None:
        """
        @param pcs: Dict[ key->tile_type, value->count ]  obtained from get_cell_potential_states_Xcardinals.
        @return: Tuple[ keys-> tile types, value-> probability ], where probability is normalized [0, 1].
            If pcs is empty then returns None, None
        """
        if not pcs:
            return None

        counts = np.array(list(pcs.values()), dtype=np.float32)
        probabilities = counts / counts.sum()

        # assert (probabilities <= 1).all(), "Probabilities must be less than or equal to 1"

        return list(pcs.keys()), probabilities

    def node_value(self, node: Node):
        return (
            # depth: can be used to prioritize nodes w/ high depth for a quicker generation
                (1 + self._number_of_tiles_to_process - node.depth()) * self._rev_depth_w +

                # cost: if temperature is high, this is the most promising locally,
                # otherwise it can be somewhat random or steer the generation towards the sample's frequencies
                node.cost() * self._node_cost_w +

                # extra: how "fuzzy" is the boundary ( unsure if useful )
                node.extra * self._ps_avg_ent_w
        )

    def successors(self, node):
        from py_search.base import Node
        """
        Generate all possible next states
        """

        if self.is_generation_aborted():
            raise InterruptedError()

        # world state is kept in self._temp_world_state; updated here when closing the node.
        # get_world_state func rollbacks any actions when depth is maintained or decreased.

        if node.depth() == 0:
            self._best_node = node
            self.open_nodes_on_depth_zero()
            world_state = self._temp_world_state
        else:
            world_state = self.get_world_state(self._last_node, node)
            self._min_temperature = self.get_new_temperature(node.depth(), self._last_node.depth())
        self._last_node = node

        depth = node.depth()
        if depth > self._best_node.depth():
            self._best_node = node
            if self._stop_and_ticker is not None:
                self._stop_and_ticker[1 + self._pid] += 1
        if depth >= self._number_of_tiles_to_process:
            self._stop = True
            print("\nEnded search with all tiles filled.")
            return

        if self._plateau_stop_steps > 0:
            self._plateau_check_ticker += 1
            if self._plateau_check_ticker >= self._plateau_stop_steps:
                if self._prev_best_depth == self._best_node.depth():
                    self._stop = True
                    print("\nEnded due to depth plateauing.")
                    if self._stop_and_ticker is not None:
                        self._stop_and_ticker[
                            1 + self._pid] += self._number_of_tiles_to_process - self._best_node.depth()
                    return
                self._plateau_check_ticker = 0
                self._prev_best_depth = self._best_node.depth()

        iyxs = self._temp_world_open_tiles

        def takewhile_and_last(predicate, iterable):
            last_element = None
            for item in iterable:
                if predicate(item):
                    yield item
                else:
                    last_element = item
                    break
            if last_element is None:
                return
            yield last_element

        potential_collapses_it = (((y, x), self.get_cell_potential_states_and_costs(y, x, world_state, depth)) for
                                  (y, x) in iyxs)
        potential_collapses = list(
            takewhile_and_last(lambda x: x[1][2] is not None and x[1][2] > 0, potential_collapses_it))
        # [ ( 0:(x, y) , 1:( 0:states, 1:costs, 2:entropy) ) ]

        # check if last is impossible
        if len(potential_collapses) == 0 or potential_collapses[-1][1][2] is None:
            node.node_cost = float("inf")  # the node has now been closed, but it could help w/ debugging
            return

        # check if last element has entropy zero, if so, it should be the only one expanded ( a collapse )
        if potential_collapses[-1][1][2] == 0:
            potential_collapses = [potential_collapses[-1]]

        # print(f"depth = {depth:5,.0f}  |  temperature={self._min_temperature:5,.1f}  |  "
        #      f"freq_depth_adjustment={self._tile_freq_adjustment_func(depth):6,.2f}  |  "
        #      f"open tiles:{len(iyxs):5,.0f}    ", end="\r")

        boundary_avg_entropy = sum(e for _, (_, _, e) in potential_collapses) / len(
            potential_collapses)  # will be lagging by one state

        for (y, x), (potential_states, costs, entropy) in potential_collapses:
            items = zip(potential_states, costs)

            # if self._temperature > xxx:  # TODO consider making this an option set by the user
            #    # items = sorted(items, key=itemgetter(1))[:2]
            #    items = sorted(items, key=itemgetter(1))
            #    take = round(2 * self._temperature / self._max_temperature + len(items) * (
            #                1 - self._temperature / self._max_temperature))
            #    items = items[:take]

            for tile_type, cost in items:
                action = ((y, x), tile_type)
                state = node.state[1] ^ hash(action)  # zobrist like
                yield Node(state=(depth + 1, state), parent=node, action=action, node_cost=cost,
                           extra=boundary_avg_entropy)

    def goal_test(self, state_node, goal_node=None):
        # state is not kept in each node, so the checks are done when closing a node.
        # goal_test is only defined to terminate the search;
        # the real goal test is done in the successors method
        return self._stop

    def get_world_state(self, last_node: Node, current_node: Node):
        start_depth = min(last_node.depth(), current_node.depth())

        p_node = last_node
        c_node = current_node

        for _ in range(p_node.depth() - start_depth):
            self.revert_action(p_node.action)
            p_node = p_node.parent

        for _ in range(c_node.depth() - start_depth):
            c_node = c_node.parent

        common_depth = 0
        for i in range(start_depth + 1)[::-1]:
            if p_node.state == c_node.state:  # hashes may collide, so albeit rare, this may trigger an error later when updating open tiles
                common_depth = i
                break
            self.revert_action(p_node.action)
            p_node = p_node.parent
            c_node = c_node.parent

        # NOTE: node is removed from opened set when closing; thus, the order of operations needs to be preserved
        c_node = current_node
        nodes_to_apply = []
        for _ in range(current_node.depth() - common_depth):
            nodes_to_apply.append(c_node)
            c_node = c_node.parent

        for node in nodes_to_apply[::-1]:
            self.apply_action(node.action)

        return self._temp_world_state

    def open_nodes_on_depth_zero(self):
        if self._starting_state is None:
            # open center tile
            self._temp_world_open_tiles.add((self._world_tdims[0] // 2, self._world_tdims[1] // 2))
            return
        # otherwise -> find all in starting state

        relative_indices_to_check = [(_y, _x) for _y in range(3) for _x in range(3) if _y != 1 or _x != 1]
        if not self._use_8cardinals:
            relative_indices_to_check = [relative_indices_to_check[i] for i in [1, 3, 4, 6]]

        wost = self._temp_world_open_tiles
        for (y, x), tile in np.ndenumerate(self._starting_state):
            if tile != 0:
                continue
            indices_to_check = [(__y, __x) for (_y, _x) in relative_indices_to_check
                                if 0 <= (__y := y + _y) < self._temp_world_state.shape[0]
                                and 0 <= (__x := x + _x) < self._temp_world_state.shape[1]]

            if any(self._starting_state[__y, __x] != 0 for (__y, __x) in indices_to_check):
                wost.add((y, x))

    def revert_action(self, node_action):
        (pos, tile_type) = node_action
        self._tile_counts[self._temp_world_state[*pos]] -= 1
        self._temp_world_state[*pos] = 0
        self.reopen_node(*pos)

    def apply_action(self, node_action):
        (pos, tile_type) = node_action
        self._temp_world_state[*pos] = tile_type
        self._tile_counts[tile_type] += 1
        self.close_node(*pos)

    def get_solution_state(self):
        node: Node = self._best_node
        encoded_state = np.zeros(self._temp_world_state.shape[:2]) if self._starting_state is None \
            else self._starting_state.copy()
        for _ in range(node.depth()):
            (y, x), tile_hash = node.action
            node = node.parent
            if tile_hash != 0:
                encoded_state[y, x] = tile_hash

        return encoded_state.astype(NP_ENCODED_TILE_TYPE)
