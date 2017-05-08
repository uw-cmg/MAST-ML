__author__ = 'haotian;Tam;Ben'
import numpy as np
import traceback
import pandas as pd


def parse(filename, weights = False, separator=','):
    """
    parse a file into parse.Data object
    :param filename: name of the file to be parsed
    :param separator: the separator used in this file, default to be ','
    :return: the parser.Data object reprenting this data
    """
    try:
        data = pd.read_csv(filename)
        features = data.columns
        return Data(features, data)
    except Exception as err:
        traceback.print_exc()
        print("an error occurred during parsing, no data object created.\n"
              "Error Message: {}".format(err))
    return None


def to_digit(x):
    """
    convert a string into an int if it represents an int
    otherwise convert it into a float if it represents a float
    otherwise do nothing and return it directly
    :param x: the input string to be converted
    :return: the result of convert
    """
    if not isinstance(x, str):
        return x
    if x == '': return None
    try:
        y = int(x)
        return y
    except ValueError:
        pass
    try:
        y = float(x)
        return y
    except ValueError:
        pass
    return x


class Data:
    def __init__(self, features, data):
        self.__features = features
        self.__data = data
        self.__filtered_data = data
        self.x_features = features[:-1]
        self.y_feature = None
        self.__max = []
        self.__min = []
        try:
            self.__calculate_data_range()
        except TypeError:
            print("Warning: error calculating data range, probably due to string types.")
            traceback.print_exc()
            print("Script will continue. data_parser normalization may not work correctly.")
        return

    def set_x_features(self, feature_list):
        if isinstance(feature_list, str):
            feature_list = [feature_list]
        for feature in feature_list:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return False
        self.x_features = feature_list
        return True

    def set_y_feature(self, feature):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        self.y_feature = feature
        #BA - figure out if we want to keep this? a special flag to filter on?
        #self.add_exclusive_filter(self.y_feature,'=','FLAG')
        #self.overwrite_data_w_filtered_data()
        return True

    # append data within self.data to self.filtered_data satisfying (operator,threshold) for a feature
    def add_inclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        # index = self.__features.index(feature)
        if self.__filtered_data.equals(self.__data):
            filtered_data_old = None
        else:
            filtered_data_old = self.__filtered_data
        if '>' in operator:
            filtered_data = self.__data[self.__data[feature] > threshold]
        elif '<' in operator:
            filtered_data = self.__data[self.__data[feature] < threshold]
        elif '=' in operator:
            filtered_data = self.__data[self.__data[feature] == threshold]
        elif '<>' in operator:
            filtered_data = self.__data[self.__data[feature] != threshold]
        elif 'contains' in operator:
            filtered_data = self.__data[self.__data[feature] == threshold]

        if filtered_data_old is None:
            filtered_data_cat = filtered_data
        else:
            filtered_data_cat = pd.concat([filtered_data,filtered_data_old]).drop_duplicates()
        self.__filtered_data = filtered_data_cat
        return True

    # remove data from self.filtered_data not satisfying (operator,threshold) for a feature
    def add_exclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        if self.__filtered_data.equals(self.__data):
            filtered_data_old = self.__data
        else:
            filtered_data_old = self.__filtered_data

        if '>' in operator:
            filtered_data = filtered_data_old[filtered_data_old[feature] > threshold]
        elif '<' in operator:
            filtered_data = filtered_data_old[filtered_data_old[feature] < threshold]
        elif '=' in operator:
            filtered_data = filtered_data_old[filtered_data_old[feature] == threshold]
        elif '<>' in operator:
            filtered_data = filtered_data_old[filtered_data_old[feature] != threshold]
        elif 'contains' in operator:
            filtered_data = filtered_data_old[filtered_data_old[feature] == threshold]

        self.__filtered_data = filtered_data
        return True

    def remove_all_filters(self):
        self.__filtered_data = self.__data
        return True

    # the equivalent of parsing a CSV of filtered data
    def overwrite_data_w_filtered_data(self):
        self.__data = self.__filtered_data

    # todo clarify, different from 's' normalization?
    def std_normalization(self, features=None):
        if features is None:
            features = [self.y_feature]
        elif isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return False
        for feature in features:
            x = self.get_data(feature)
            x = [i for i in x[feature]]
            result = []
            for i in x:
                if i is not None:
                    result.append(i)
            avg = sum(result) / len(result)
            sqr_err = 0
            for i in result:
                sqr_err += (i - avg) ** 2
            sqr_err /= len(result)
            std_err = sqr_err ** 0.5
            result = []
            for i in x:
                if i is None:
                    result.append(None)
                else:
                    result.append((i - avg) / std_err)
            self.add_feature('std_N_{}'.format(feature), result)

    def normalization(self, features=None, normalization_type='s'):
        if features is None:
            features = self.x_features
        elif isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return False
            if self.__max[self.__features.index(feature)] is None:
                print("Feature [{}] is not numerical".format(feature))
                return False
        if normalization_type == 's':
            for feature in features:
                result = []
                index = self.__features.index(feature)
                cur_max = self.__max[index]
                cur_min = self.__min[index]
                if cur_max is None:
                    print('feature[{}] max is none'.format(feature))
                elif cur_min is None:
                    print('feature[{}] min is none'.format(feature))
                for line in self.__data:
                    if line[index] is None:
                        result.append(None)
                    else:
                        result.append((line[index] - cur_min) / (cur_max - cur_min))
                self.add_feature('N_{}'.format(feature), result)
        elif normalization_type == 't':
            all_max = max([self.__max[x] for x in range(len(self.__features)) if self.__features[x] in features])
            all_min = min([self.__min[x] for x in range(len(self.__features)) if self.__features[x] in features])
            for feature in features:
                result = []
                index = self.__features.index(feature)
                self.__max[index] = all_max
                self.__min[index] = all_min
                for line in self.__data:
                    if line[index] is None:
                        result.append(None)
                    else:
                        result.append((line[index] - all_min) / (all_max - all_min))
                self.add_feature('N_{}'.format(feature), result)
        else:
            print("unknown normalization_type '{}'; "
                  "expect 's' for separate or 't' for together".format(normalization_type))
            return False
        return True

    def unnormalization_data_point(self, feature, data):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return None
        index = self.__features.index(feature)
        cur_max = self.__max[index]
        cur_min = self.__min[index]
        return data*(cur_max-cur_min)+cur_min

    def get_data(self, features=None):
        if isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return None
        output = self.__filtered_data[features]
        return output

    def get_x_data(self):
        assert (self.y_feature != None), "Must set y feature first before getting x"
        return self.get_data(features=self.x_features)

    def get_y_data(self):
        return self.get_data(features=self.y_feature)

    def __calculate_data_range(self):
        self.__max = self.__data.max
        self.__min = self.__data.min

    def output(self, filename, features=None, datatype='a'):
        if features is None:
            features = self.__features
        else:
            if isinstance(features, str):
                features = [features]
            for feature in features:
                if feature not in self.__features:
                    print("can't find [{}] in features, no file is created".format(feature))
                    return False
        if datatype == 'a':
            data = self.__data
        elif datatype == 'f':
            data = self.__filtered_data
        else:
            print("can't recognize data [{}], please pass in 'a' for all data, or 'f' for filtered data".format(data))
            return False
        f = open(filename, 'w')
        index_list = [self.__features.index(feature) for feature in features]
        for i in range(len(index_list)):
            if i != len(index_list) - 1:
                f.write('{},'.format(features[i]))
            else:
                f.write('{}\n'.format(features[i]))
        for line in data:
            for i in range(len(index_list)):
                if i != len(index_list) - 1:
                    f.write('{},'.format(line[index_list[i]]))
                else:
                    f.write('{}\n'.format(line[index_list[i]]))
        f.close()
        return True

    def add_feature(self, name, values):
        if len(value) != len(self.__data):
            print('unmatch data length')
            return False
        self.__features.append(name)
        self.__Data = pd.Series(values, index=self.__Data.index)
        print('feature [{}] added'.format(name))
        self.__calculate_data_range()
        return True

    def normalize_feature_global_minmax(self, featurename="",
                                        gmin=None, gmax=None, newname=""):
        """Normalize a feature using predetermined minimum and maximum values.
            Use for normalizing across multiple data_parser Data objects.
        """
        if featurename not in self.__features:
            print("can't find [{}] in features".format(featurename))
            return False
        if newname == "":
            newname = "N_{}".format(featurename)
        result = (self.__data[featurename]-gmin)/(gmax-gmin)
        self.add_feature(newname, result)
        return True

    def add_log10_feature(self, featurename="", newname=""):
        if featurename not in self.__features:
            print("can't find [{}] in features".format(featurename))
            return False
        if newname == "":
            newname = "log10_{}".format(featurename)
        result = np.log10(self.__data[featurename])
        self.add_feature(newname, result)
        return True
