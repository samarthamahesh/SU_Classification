import random
import numpy as np


class AdultDataset():

    def __init__(self, names_path="../datasets/Adult/adult.names", data_path="../datasets/Adult/adult.data"):
        self.lab2ind, self.ind2lab, self.map, self.inv_map, self.out_map, self.inv_out_map = self.get_premaps(names_path)
        self.X, self.Y = self.get_data(data_path, 0)

    def get_gen0(self, n):
        X0 = self.X[self.Y == 0]
        n0 = len(X0)
        ind = random.sample(range(n0), n)
        return X0[ind]

    def get_gen1(self, n):
        X1 = self.X[self.Y == 1]
        n1 = len(X1)
        ind = random.sample(range(n1), n)
        return X1[ind]

    def get_premaps(self, names_path):
        mapping, lab2ind, inv_mapping, ind2lab = {}, {}, {}, {}

        # Mapping dictionary to map different labels in different attributes to indices
        for i, line in enumerate(open(names_path, 'r').read().split('\n')[-15:-1]):
            lab, arr = [x.strip() for x in line.split(":")]
            arr = arr[:-1]
            lab2ind[lab] = i

            if arr != 'continuous':
                mapping[i] = {}
                vals = [x.strip() for x in arr.split(',')]

                for j, val in enumerate(vals):
                    mapping[i][val] = j+1

        # Mapping dictionary to map indices to label names
        for i in mapping:
            inv_mapping[i] = {mapping[i][j]: j for j in mapping[i]}

        # Mapping for attributes
        ind2lab = {lab2ind[i]: i for i in lab2ind}

        # Mapping for output labels
        out_map = {
            '>50K': 1,
            '<=50K': 0
        }

        # Inverse mapping output labels
        inv_out_map = {out_map[i]: i for i in out_map}

        return lab2ind, ind2lab, mapping, inv_mapping, out_map, inv_out_map



    def get_data(self, data_path, noise):
        data_x = []
        data_y = []

        for line in open(data_path, 'r').read().split('\n'):
            if line == "":
                continue
            
            dat = [x.strip() for x in line.split(',')]

            tmp = []
            incomplete = False
            for i, val in enumerate(dat[:-1]):
                if val == '?':
                    incomplete = True
                    break

                if i in self.map:
                    tmp.append(float(self.map[i][val]))
                else:
                    tmp.append(float(val))

            if incomplete:
                continue

            data_x.append(np.array(tmp))
            data_y.append(self.out_map[dat[-1]])

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        n = len(data_y)
        ind = random.sample(list(range(n)), int(n*noise))
        data_y[ind] = 1 - data_y[ind]

        return data_x, data_y