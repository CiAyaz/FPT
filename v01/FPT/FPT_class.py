from collections import defaultdict
from gettext import translation
from hmac import trans_36
from importlib.resources import path
from os import XATTR_SIZE_MAX
from matplotlib.pyplot import get
import numpy as np
from FPT.tools import *


class FPT():
    """Main class for computing passage times distributions.
    Computes passage times: first first passage times, all first passage times and transition path times.
    trajectories: 1d numpy array trajectory or a list of strings with paths to 1d trajectories,
    dt: time step,
    xstart_vector and xfinal_vector: 1d arrays with positions to compute the times it takes to go from xstart to xfinal,
    array_size: the size of the passage times arrays. As an estimate, use expected maximum number of passage events."""

    def __init__(self, 
    trajectories,
    dt, 
    xstart_vector, 
    xfinal_vector, 
    x_range = None, 
    array_size=int(1e6), 
    savefiles=False, 
    path_for_savefiles='./',
    file_name='',
    nbins=100):
        self.trajectories = trajectories
        self.dt = dt
        self.xstart_vector = xstart_vector
        self.xfinal_vector = xfinal_vector
        self.x_range = x_range
        self.array_size = array_size
        self.savefiles = savefiles
        self.path_for_savefiles = path_for_savefiles
        self.file_name = file_name
        # this is what we use for continuation of calc
        self._integer_variables = np.zeros(2, dtype=np.int64)
        self._float_variables = np.zeros(4)
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
        self.total_trajectory_length = 0
        self.nbins = nbins
        self.transition_path_len = 0
        self.time_distr_nbins = 100
        self.comp_time_distr = True

    def parse_input(self):
        """trajectories can be str or np.ndarray or a list of such.
        xstart_vector and xfinal_vector must a list of numerical values or numpy ndarray."""
        if not isinstance(self.trajectories, list):
            self.trajectories = [self.trajectories]
        if not all([isinstance(traj, (str, np.ndarray)) for traj in self.trajectories]):
            raise TypeError("All trajectories should be str or numpy array.")
        self.number_of_trajs = len(self.trajectories)
        if not isinstance(self.xstart_vector, (list, np.ndarray)):
            self.xstart_vector = [self.xstart_vector]
        if not isinstance(self.xfinal_vector, (list, np.ndarray)):
            self.xfinal_vector = [self.xfinal_vector]
        self.number_xstarts = len(self.xstart_vector)
        self.number_xfinals = len(self.xfinal_vector)
        if self.file_name =='':
            pass
        else:
            if not self.file_name[0] == '_':
                self.file_name = '_' + self.file_name


    def check_xfinal_reached(self):
        if self._integer_variables[1] != 0:
            nonzero_length = len(self.fpt_array_with_recrossings)
            self.fpt_dummy = self.fpt_array_with_recrossings[-self._integer_variables[1] :]
            self.fpt_array_with_recrossings = self.fpt_array_with_recrossings[
                : nonzero_length - self._integer_variables[1]
            ]
        else:
            self.fpt_dummy = np.array([])

    def compute_single_passage_time(self, x, xstart, xfinal):

        sign_x_minus_xstart = np.sign(x - xstart)
        sign_x_minus_xfinal = np.sign(x - xfinal)
        (
            self._float_variables,
            self._integer_variables,
            self.fpt_array,
            self.tpt_array,
            self.fpt_array_with_recrossings
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
        self.fpt_array_with_recrossings = np.trim_zeros(self.fpt_array_with_recrossings, "b")
        if self._float_variables[-1] == 0.:
            self.fpt_array_with_recrossings = np.append(self.fpt_array_with_recrossings, 0.)
        self.fpt_array = np.trim_zeros(self.fpt_array, "b")
        self.tpt_array = np.trim_zeros(self.tpt_array, "b")

    def get_data(self, traj):
        if isinstance(traj, str):
            return np.load(traj)
        return traj

    def compute_time_distributions(self):
        self.fpt_distr = dict()
        self.tpt_distr = dict()
        self.fpt_wr_distr = dict()
        output_dictionaries = [self.fpt_distr, self.tpt_distr, self.fpt_wr_distr]
        input_dictionaries = [self.fpt_dict, self.tpt_dict, self.fpt_wr_dict]
        for dict_out, dict_in in zip(output_dictionaries, input_dictionaries):
            for key in dict_in.keys():
                dict_length = len(dict_in[key])
                if dict_length == 0:
                    dict_out[key] = np.array([])
                elif dict_length == 1:
                    dict_out[key] = np.array([dict_in[key][0], 1.])
                else:
                    key_edges = np.linspace(dict_in[key].min(), dict_in[key].max(), self.time_distr_nbins)
                    key_bw = 1.06 * dict_length ** (-1/5) * np.std(dict_in[key])
                    p = kde_epanechnikov(dict_in[key], key_edges, key_bw, norm=dict_length)
                    dict_out[key] = np.concatenate((key_edges[:,None], p[:,None]), axis=1)
                    

    def compute_passage_time_arrays(self):
        self.parse_input()
        self.fpt_array_with_recrossings = np.zeros((self.array_size,), dtype=np.float64)
        print('computing times array')
        for traj in self.trajectories:
            x = self.get_data(traj)
            for index_xstart, xstart in enumerate(self.xstart_vector):
                for index_xfinal, xfinal in enumerate(self.xfinal_vector):
                    if xstart == xfinal:
                        continue
                    else:
                        self.compute_single_passage_time(x, xstart, xfinal)

                        self.check_xfinal_reached()

                        self.fpt_wr_dict[f"{xstart:.2f}_{xfinal:.2f}"] = np.append(
                            self.fpt_wr_dict[f"{xstart:.2f}_{xfinal:.2f}"],
                            self.fpt_array_with_recrossings,
                        )
                        self.fpt_dict[f"{xstart:.2f}_{xfinal:.2f}"] = np.append(
                            self.fpt_dict[f"{xstart:.2f}_{xfinal:.2f}"], self.fpt_array
                        )
                        self.tpt_dict[f"{xstart:.2f}_{xfinal:.2f}"] = np.append(
                            self.tpt_dict[f"{xstart:.2f}_{xfinal:.2f}"], self.tpt_array
                        )
                        self.fpt_array_with_recrossings = np.zeros((self.array_size,), dtype=np.float64)
                        self.fpt_array_with_recrossings[: self._integer_variables[1]] = self.fpt_dummy

        if self.comp_time_distr:
            print('computing times distributions')
            self.compute_time_distributions()

        if self.savefiles:
            print("saving output arrays!")
            np.save(self.path_for_savefiles+'fpt_dict'+self.file_name, dict(self.fpt_dict))
            np.save(self.path_for_savefiles+'tpt_dict'+self.file_name, dict(self.tpt_dict))
            np.save(self.path_for_savefiles+'fpt_with_recrossings_dict'+self.file_name, dict(self.fpt_wr_dict))
            np.save(self.path_for_savefiles+'fpt_distr'+self.file_name, self.fpt_distr)
            np.save(self.path_for_savefiles+'tpt_distr'+self.file_name, self.tpt_distr)
            np.save(self.path_for_savefiles+'fpt_with_recrossings_distr'+self.file_name, self.fpt_wr_distr)

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
        if self.transition_path_indices[0,0] == 0 and isinstance(self.x_dummy, np.ndarray):
           self.transition_paths = np.append(self.transition_paths, self.x_dummy)
           self.transition_path_len += len(self.x_dummy)
        if self.transition_path_indices[-1,1] != 0:
            for index in range(len(self.transition_path_indices)):
                start_index = self.transition_path_indices[index, 0]
                end_index = self.transition_path_indices[index, 1]
                self.transition_path_len += (end_index - start_index)
                self.transition_paths = np.append(self.transition_paths, x[start_index: end_index+1])
        else:
            for index in range(len(self.transition_path_indices)-1):
                start_index = self.transition_path_indices[index, 0]
                end_index = self.transition_path_indices[index, 1]
                self.transition_path_len += (end_index - start_index)
                self.transition_paths = np.append(self.transition_paths, x[start_index: end_index+1])
            self.x_dummy = x[self.transition_path_indices[-1,0]:]

    def compute_x_range_and_bw(self):
        print('Estimating range in total trajectory and bandwidth for kde')
        if not isinstance(self.trajectories, list):
            self.parse_input()
        if len(self.trajectories) > 1:
            xmax = None
            xmin = None
            indices = np.random.randint(
                low=0, 
                high=len(self.trajectories) - 1, 
                size=int(len(self.trajectories) / 10) + 1)
            trajs = []
            for ind in indices:
                trajs.append(self.trajectories[ind])
            xmean = 0.
            for traj in trajs:
                x = self.get_data(traj)
                xmax_test = np.max(x)
                xmin_test = np.min(x)
                if xmax == None or xmax < xmax_test:
                    xmax = xmax_test
                if xmin == None or  xmin > xmin_test:
                    xmin = xmin_test
                xmean += np.mean(x)
            xmean /= len(trajs)
            xvar = 0.
            for traj in trajs:
                x = self.get_data(traj)
                xdiff = (x - xmean) **2 / len(x)
                xvar += np.sum(xdiff)
            xvar /= len(trajs)

            self.x_range = (xmin, xmax)
            self.bw = 1.06 * (len(trajs) * len(x)) **(-1/5) * np.sqrt(xvar)
        else:
            x = self.get_data(self.trajectories[0])
            self.x_range = (np.min(x), np.max(x))
            self.bw = 1.06 * len(x) **(-1/5) * np.std(x)
        print('estimated range in total trajectory is ', self.x_range)
        print('estimated bw in total trajectory is ', self.bw)

    def comp_edges(self):
        xstart = self.xstart_vector[0]
        xfinal = self.xfinal_vector[0]
        x_left = min(xstart, xfinal)
        x_right = max(xstart, xfinal)

        self.PxTP_edges = np.linspace(x_left, x_right, self.nbins)
        self.Px_edges = np.linspace(x_left, x_right, self.nbins)
        width = self.Px_edges[1]-self.Px_edges[0]

        x_left -= width
        while x_left > self.x_range[0]:
            self.Px_edges = np.append(self.Px_edges[::-1], x_left)[::-1]
            x_left -= width
        x_right += width
        while x_right < self.x_range[1]:
            self.Px_edges = np.append(self.Px_edges, x_right)
            x_right += width
        self.nbins_Px = len(self.Px_edges)

        
    def compute_Px(self, x):
        p = kde_epanechnikov(x, self.Px_edges, self.bw)
        return p

    def compute_PxTP(self, x):
        p = kde_epanechnikov(x, self.PxTP_edges, self.bw)
        return p

    def write_info_file(self):
        with open(self.path_for_savefiles+'info.txt', 'a') as f:
                f.write('estimated range in total trajectory is (%.4g, %.4g)'%(self.x_range[0],self.x_range[1]))
                f.write('\n')
                f.write('estimated bw in total trajectory is %.4g'%self.bw)
                f.write('\n')
                f.write('length of transition paths is %d'%self.transition_path_len)
                f.write('\n')
                f.write('total number of transitions is %d'%self._integer_variables_TPT[1])


    def compute_PTPx(self):
        """
        Compute transition path probability
        """
        self.parse_input()
        self.transition_paths = np.array([])
        self.PxTP = np.zeros(self.nbins)
        if self.x_range == None:
            self.compute_x_range_and_bw()
        self.comp_edges()
        self.Px = np.zeros(self.nbins_Px)

        print('Computing PTPx')
        for traj in self.trajectories:
            x = self.get_data(traj)

            self.compute_transition_paths(x)

            self.concatenate_transition_paths(x)

            if len(self.transition_paths) != 0:
                PxTP_dummy = self.compute_PxTP(self.transition_paths)
                self.PxTP += PxTP_dummy
                self.transition_paths = np.array([])

            Px_dummy = self.compute_Px(x)

            self.Px += Px_dummy
            self.total_trajectory_length += len(x)

        self.PxTP = self.PxTP/np.trapz(self.PxTP, self.PxTP_edges)
        self.Px = self.Px/np.trapz(self.Px, self.Px_edges)

        print("length of transition paths is ", self.transition_path_len)

        self.PTP = self.transition_path_len / self.total_trajectory_length

        inds = np.where(np.logical_and(
            self.Px_edges - self.PxTP_edges[0] >= 0,
            self.Px_edges - self.PxTP_edges[-1] <= 0
            ))[0]
        self.PTPx = self.PTP * self.PxTP / self.Px[inds]

        if self.savefiles:
            self.PTPx = np.concatenate((
                self.PxTP_edges[:, None], 
                self.PTPx[:, None]), 
                axis = 1)

            print("saving output arrays!")
            np.save(self.path_for_savefiles+'PTP'+self.file_name, np.array([self.PTP]))
            np.save(self.path_for_savefiles+'PxTP'+self.file_name, self.PxTP)
            np.save(self.path_for_savefiles+'Px'+self.file_name, self.Px)
            np.save(self.path_for_savefiles+'PTPx'+self.file_name, self.PTPx)

            self.write_info_file()