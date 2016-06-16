__author__ = 'haotian'
import numpy as np

def parse(filename, separator=','):
    """
    parse a file into parse.Data object
    :param filename: name of the file to be parsed
    :param separator: the separator used in this file, default to be ','
    :return: the parser.Data object reprenting this data
    """
    try:
        f = open(filename, 'r')
        features = f.readline()[:-1].split(separator)
        data = f.read().splitlines()
        f.close()
        itr = 0
        for line in data:
            line = line.split(separator)
            for i in range(len(line)):
                line[i] = to_digit(line[i])
            data[itr] = line
            itr += 1
        return Data(features, data)
    except Exception as err:
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
        self.__filtered_data = list(data)
        self.__x_features = features[:-1]
        self.__y_feature = features[-1]
        self.__max_min = []
        return

    def set_x_features(self, feature_list):
        if isinstance(feature_list, str):
            feature_list = [feature_list]
        for feature in feature_list:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return False
        self.__x_features = feature_list
        return True

    def set_y_feature(self, feature):
        if feature not in self.__features:
            print("can't find [] in features".format(feature))
            return False
        self.__y_feature = feature
        return True

    #include features satisfying (operator,threshold) in filtered_data
    def add_inclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        index = self.__features.index(feature)
        if self.__filtered_data == self.__data: filtered_data = []
        else: filtered_data = self.__filtered_data
        for line in self.__data:
            if line[index] > threshold and '>' in operator:
                filtered_data.append(line)
            elif line[index] == threshold and '=' in operator:
                filtered_data.append(line)
            elif line[index] < threshold and '<' in operator:
                filtered_data.append(line)
            elif str(threshold) in str(line[index]) and 'contains' in operator:
                filtered_data.append(line)
        self.__filtered_data = filtered_data
        return True

    #do not include features satisfying (operator,threshold) in filtered_data
    def add_exclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        index = self.__features.index(feature)
        filtered_data = list(self.__filtered_data)
        for line in self.__data:
            if line[index] > threshold and '>' in operator:
                filtered_data.remove(line)
            elif line[index] == threshold and '=' in operator:
                filtered_data.remove(line)
            elif line[index] < threshold and '<' in operator:
                filtered_data.remove(line)
            elif str(threshold) in str(line[index]) and 'contains' in operator:
                filtered_data.remove(line)
        self.__filtered_data = filtered_data
        return True

    def remove_all_filters(self):
        self.__filtered_data = list(self.__data)
        return True

    #the equivalent of parsing a CSV of filtered data
    def set_data_to_filtered_data(self):
        self.__data = self.__filtered_data
        self.__filtered_data = self.__filtered_data

    def normalization(self, features=None, normalization_type='s'):
        if features is None:
            features = self.__x_features
        elif isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return False
            if self.__max_min[0][self.__features.index(feature)] is None:
                print("Feature [{}] is not numerical".format(feature))
                return False
        if normalization_type == 's':
            for feature in features:
                index = self.__features.index(feature)
                cur_max = self.__max_min[0][index]
                cur_min = self.__max_min[1][index]
                for line in self.__data:
                    line[index] = (line[index]-cur_min)/(cur_max-cur_min)
                for line in self.__filtered_data:
                    line[index] = (line[index]-cur_min)/(cur_max-cur_min)
        elif normalization_type == 't':
            all_max = max([self.__max_min[0][x] for x in range(len(self.__features)) if self.__features(x) in features])
            all_min = min([self.__max_min[1][x] for x in range(len(self.__features)) if self.__features(x) in features])
            for feature in features:
                index = self.__features.index(feature)
                self.__max_min[0][index] = all_max
                self.__max_min[1][index] = all_min
                for line in self.__data:
                    line[index] = (line[index]-all_min)/(all_max-all_min)
                for line in self.__filtered_data:
                    line[index] = (line[index]-all_min)/(all_max-all_min)
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
        cur_max = self.__max_min[0][index]
        cur_min = self.__max_min[1][index]
        return data*(cur_max-cur_min)+cur_min

    def get_data(self, features=None):
        if features is None:
            features = self.__features
        if isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return None
        output = []
        index_list = [i for i in range(len(self.__features)) if self.__features[i] in features]
        for line in self.__filtered_data:
            output.append([line[i] for i in index_list])
        return output

    def get_x_data(self):
        return self.get_data(features=self.__x_features)

    def get_y_data(self):
        return self.get_data(features=self.__y_feature)

    def __calculate_data_range(self):
        maxes = list(self.__data[0])
        mins = list(self.__data[0])
        for line in self.__data:
            for i in range(len(line)):
                if isinstance(line[i], str):
                    continue
                if line[i] > maxes[i]:
                    maxes[i] = line[i]
                elif line[i] < mins[i]:
                    mins[i] = line[i]
        for i in range(len(self.__features)):
            if isinstance(maxes[i], str):
                maxes[i] = None
                mins[i] = None
        self.__max_min.append(maxes)
        self.__max_min.append(mins)
