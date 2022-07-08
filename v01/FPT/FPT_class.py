from collections import defaultdict
from hmac import trans_36
from importlib.resources import path
from os import XATTR_SIZE_MAX
import numpy as np
from FPT.tools import compute_passage_times, find_transition_paths


class FPT():
    """Main class for computing passage times distributions.
    Computes passage times: first first passage times, all first passage times and transition path times.
    trajectories: 1d numpy array trajectory or a list of strings with paths to 1d trajectories,
    dt: time step,
    xstart_vector and xfinal_vector: 1d arrays with positions to compute the times it takes to go from xstart to xfinal,
    array_size: the size of the passage times arrays. As an estimate, use expected maximum number of passage events."""

    def __init__(self, dt, xstart_vector, xfinal_vector, x_len, x_range = None, array_size=int(1e6), savefiles=False, path_for_savefiles='./'):
        self.dt = dt
        self.xstart_vector = xstart_vector
        self.xfinal_vector = xfinal_vector
        self.x_range = x_range
        self.array_size = array_size
        self.savefiles = savefiles
        self.path_for_savefiles = path_for_savefiles
        # this is what we use for continuation of calc
        self._integer_variables = np.zeros(2, dtype=np.int64)
        self._float_variables = np.zeros(3)
        self._integer_variables_TPT = np.zeros(3, dtype=np.int64)
        self._float_variables_TPT = np.zeros(2)
        # fields for the results
        self.fpt_wr_dict = defaultdict(lambda: np.array([]))
        self.fpt_dict = defaultdict(lambda: np.array([]))
        self.tpt_dict = defaultdict(lambda: np.array([]))
        self.number_of_trajs = None
        self.number_xstarts = None
        self.number_xfinals = None
        self.fpt_array_with_recrossings = None
        self.fpt_array = None
        self.tpt_array = None
        self.transition_path_indices = None
        self.transition_paths = None
        self.x_dummy = None
        self.Px = None
        self.PxTP = None
        self.PTP = None
        self.PTPx = None
        self.Px_edges = None
        self.total_trajectory_length = x_len

    def parse_input(self, trajectories):
        """trajectories can be str or np.ndarray or a list of such.
        xstart_vector and xfinal_vector must a list of numerical values or numpy ndarray."""
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        if not all([isinstance(traj, (str, np.ndarray)) for traj in trajectories]):
            raise TypeError("All trajectories should be str or numpy array.")
        self.number_of_trajs = len(trajectories)
        if not isinstance(self.xstart_vector, (list, np.ndarray)):
            self.xstart_vector = [self.xstart_vector]
        if not isinstance(self.xfinal_vector, (list, np.ndarray)):
            self.xfinal_vector = [self.xfinal_vector]
        self.number_xstarts = len(self.xstart_vector)
        self.number_xfinals = len(self.xfinal_vector)
        return trajectories

    def check_xfinal_reached(self):
        if self._integer_variables[1] != 0:
            nonzero_length = len(self.fpt_array_with_recrossings)
            self.fpt_dummy = self.fpt_array_with_recrossings[-self._integer_variables[1] :]
            self.fpt_array_with_recrossings = self.fpt_array_with_recrossings[
                : nonzero_length - self._integer_variables[1]
            ]
        else:
            self.fpt_dummy = np.zeros(1)

    def compute_single_passage_time(self, x, xstart, xfinal):
        if xstart == xfinal:
            return
        else:
            sign_x_minus_xstart = np.sign(x - xstart)
            sign_x_minus_xfinal = np.sign(x - xfinal)
            (
                self._float_variables,
                self._integer_variables,
                self.fpt_array,
                self.tpt_array,
                self.fpt_array_with_recrossings,
                self.transition_path_indices
            ) = compute_passage_times(
                x,
                self.dt,
                sign_x_minus_xstart,
                sign_x_minus_xfinal,
                self.fpt_array_with_recrossings,
                xstart,
                xfinal,
                self._float_variables,
                self._integer_variables,
            )
        self.fpt_array_with_recrossings = self.fpt_array_with_recrossings[
            self.fpt_array_with_recrossings != 0.0
        ]
        self.fpt_array = self.fpt_array[self.fpt_array != 0.0]
        self.tpt_array = self.tpt_array[self.tpt_array != 0.0]

    def get_data(self, traj):
        if isinstance(traj, str):
            return np.load(traj)
        return traj

    def compute_passage_time_arrays(self, trajectories):
        trajectories= self.parse_input(trajectories)
        self.fpt_array_with_recrossings = np.zeros((self.array_size,), dtype=np.float64)
        for traj in trajectories:
            x = self.get_data(traj)
            for index_xstart, xstart in enumerate(self.xstart_vector):
                for index_xfinal, xfinal in enumerate(self.xfinal_vector):

                    self.compute_single_passage_time(x, xstart, xfinal)

                    self.check_xfinal_reached()

                    self.fpt_wr_dict[index_xstart, index_xfinal] = np.append(
                        self.fpt_wr_dict[index_xstart, index_xfinal],
                        self.fpt_array_with_recrossings,
                    )
                    self.fpt_dict[index_xstart, index_xfinal] = np.append(
                        self.fpt_dict[index_xstart, index_xfinal], self.fpt_array
                    )
                    self.tpt_dict[index_xstart, index_xfinal] = np.append(
                        self.tpt_dict[index_xstart, index_xfinal], self.tpt_array
                    )
                    self.fpt_array_with_recrossings = np.zeros((self.array_size,), dtype=np.float64)
                    self.fpt_array_with_recrossings[: self._integer_variables[1]] = self.fpt_dummy

        if self.savefiles:
            print("saving output arrays!")
            np.save(self.path_for_savefiles+'fpt_dict', dict(self.fpt_dict))
            np.save(self.path_for_savefiles+'tpt_dict', dict(self.tpt_dict))
            np.save(self.path_for_savefiles+'fpt_with_recrossings_dict', dict(self.fpt_wr_dict))

    def compute_transition_paths(self, x):
        xstart = self.xstart_vector[0]
        xfinal = self.xfinal_vector[-1]
        if xstart == xfinal:
            return
        else:
            sign_x_minus_xstart = np.sign(x - xstart)
            sign_x_minus_xfinal = np.sign(x - xfinal)
            (
                self._float_variables_TPT,
                self._integer_variables_TPT,
                self.transition_path_indices
            ) = find_transition_paths(
                x,
                sign_x_minus_xstart,
                sign_x_minus_xfinal,
                self.array_size,
                self._float_variables_TPT,
                self._integer_variables_TPT
                )
        self.transition_path_indices = self.transition_path_indices[
            np.unique(np.nonzero(self.transition_path_indices)[0])]
        
        
    
    def concatenate_transition_paths(self, x):
        if len(self.transition_path_indices) == 0:
            return
        if self.transition_path_indices[0,0] == 0 and self.x_dummy != None:
           self.transition_paths = np.append(self.transition_paths, self.x_dummy)
        if self.transition_path_indices[-1,1] != 0:
            for index in range(len(self.transition_path_indices)):
                start_index = self.transition_path_indices[index, 0]
                end_index = self.transition_path_indices[index, 1]
                self.transition_paths = np.append(self.transition_paths, x[start_index: end_index+1])
        else:
            for index in range(len(self.transition_path_indices)-1):
                start_index = self.transition_path_indices[index, 0]
                end_index = self.transition_path_indices[index, 1]
                self.transition_paths = np.append(self.transition_paths, x[start_index: end_index+1])
            self.x_dummy = x[self.transition_path_indices[-1,0]:]

    def compute_x_range(self, trajectories):
        print('Computing range in total trajectory')
        xmax = None
        xmin = None
        if not isinstance(trajectories, list):
            trajectories = self.parse_input(trajectories)
        for traj in trajectories:
            x = self.get_data(traj)
            xmax_test = np.max(x)
            xmin_test = np.min(x)
            if xmax == None or xmax < xmax_test:
                xmax = xmax_test
            if xmin == None or  xmin > xmin_test:
                xmin = xmin_test
        self.x_range = (xmin, xmax)
        print('Range in total trajectory is ', self.x_range)
        


    def compute_TPT(self, trajectories, nbins=100):
        """
        Compute transition path probability
        """
        trajectories = self.parse_input(trajectories)
        self.transition_paths = np.array([])
        self.Px = np.zeros((nbins, ))
        if self.x_range == None:
            self.compute_x_range(trajectories)
        print('Computing PTPx')
        for traj in trajectories:
            x = self.get_data(traj)
            
            self.compute_transition_paths(x)

            self.concatenate_transition_paths(x)

            self.Px, self.Px_edges = np.histogram(
            x,
            bins = nbins,
            range = self.x_range)

            self.Px += self.Px

        self.Px = self.Px/np.trapz(self.Px, self.Px_edges[:-1])

        self.PxTP, self.Px_edges = np.histogram(
            self.transition_paths,
            bins = nbins,
            range = self.x_range,
            density = True)

        self.PTP = len(self.transition_paths) / self.total_trajectory_length

        self.PTPx = self.PTP * self.PxTP / self.Px