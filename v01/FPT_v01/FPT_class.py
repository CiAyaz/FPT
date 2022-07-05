from collections import defaultdict
import numpy as np
from tools import compute_passage_times


class FPT():
    """Main class for computing passage times distributions. 
    Computes passage times: first first passage times, all first passage times and transition path times.
    trajectories: 1d numpy array trajectory or a list of strings with paths to 1d trajectories,
    dt: time step,
    xstart_vector and xfinal_vector: 1d arrays with positions to compute the times it takes to go from xstart to xfinal,
    array_size: the size of the passage times arrays. As an estimate, use expected maximum number of passage events."""
    
    def __init__(self, dt, array_size = int(1e6)):
        self.dt = dt
        self.array_size = array_size
        # this is what we use for continuation of calc
        self._integer_variables = np.array([0,0], dtype = np.int64)
        self._float_variables = np.zeros(3)
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

    def parse_input(self, trajectories, xstart_vector, xfinal_vector):
        """ trajectories can be str or np.ndarray or a list of such.
            xstart_vector and xfinal_vector must a list of numerical values or numpy ndarray."""
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        if not all([isinstance(traj, (str, np.ndarray)) for traj in trajectories]):
            raise TypeError("All trajectories should be str or numpy array.")
        self.number_of_trajs = len(trajectories)
        if not isinstance(xstart_vector, (list, np.ndarray)):
            xstart_vector = [xstart_vector]
        if not isinstance(xfinal_vector, (list, np.ndarray)):
            xfinal_vector = [xfinal_vector]
        self.number_xstarts = len(xstart_vector)
        self.number_xfinals = len(xfinal_vector)
        return trajectories, xstart_vector, xfinal_vector

    def check_xfinal_reached(self):
        if self._integer_variables[1] != 0:
            nonzero_length = len(self.fpt_array_with_recrossings)
            self.fpt_dummy = self.fpt_array_with_recrossings[ - self._integer_variables[1]: ]
            self.fpt_array_with_recrossings = self.fpt_array_with_recrossings[ :nonzero_length - self._integer_variables[1] ]
        else:
            self.fpt_dummy = np.zeros(1)

    def compute_single_passage_time(self, x, xstart, xfinal):
        if xstart == xfinal:
            continue
        else:
            sign_x_minus_xstart = np.sign(x - xstart)
            sign_x_minus_xfinal = np.sign(x - xfinal)
            self._float_variables, self._integer_variables, self.fpt_array, self.tpt_array, self.fpt_array_with_recrossings = compute_passage_times(x, self.dt, sign_x_minus_xstart, sign_x_minus_xfinal, self.fpt_array_with_recrossings, xstart, xfinal, self._float_variables, self._integer_variables)
        self.fpt_array_with_recrossings = self.fpt_array_with_recrossings[self.fpt_array_with_recrossings != 0.]
        self.fpt_array = self.fpt_array[self.fpt_array != 0.]
        self.tpt_array = self.tpt_array[self.tpt_array != 0.]

    def get_data(self, traj):
        if isinstance(traj, str):
            x = np.load(traj)
        else:
            x = traj
        return x


    def compute_passage_time_arrays(self, trajectories, xstart_vector, xfinal_vector):
        trajectories, xstart_vector, xfinal_vector = self.parse_input(self, trajectories, xstart_vector, xfinal_vector)
        self.fpt_array_with_recrossings = np.zeros((self.array_size, ), dtype=np.float64)
        for traj in trajectories:
            x = self.get_data(self, traj)
            for index_xstart, xstart in enumerate(xstart_vector):
                for index_xfinal, xfinal in enumerate (xfinal_vector):

                    self.compute_single_passage_time(self, x, xstart, xfinal)

                    self.check_xfinal_reached(self)
                    
                    self.fpt_wr_dict[index_xstart, index_xfinal] = np.append(self.fpt_wr_dict[index_xstart, index_xfinal] , self.fpt_array_with_recrossings)
                    self.fpt_dict[ index_xstart, index_xfinal] = np.append(self.fpt_dict[index_xstart, index_xfinal ], self.fpt_array)
                    self.tpt_dict[ index_xstart, index_xfinal] = np.append(self.tpt_dict[index_xstart, index_xfinal ], self.tpt_array)
                    self.fpt_array_with_recrossings = np.zeros((self.array_size, ), dtype=np.float64)
                    self.fpt_array_with_recrossings[ :current_number_recrossings ] = self.fpt_dummy