__author__ = 'haotian'
import numpy as np
import traceback
def parse(filename, weights = False, separator=','):
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
        weighted_data = []
        f.close()
        itr = 0
        for line in data:
            line = line.split(separator)
            for i in range(len(line)):
                line[i] = to_digit(line[i])
            if weights and 'weight' in features:
                for i in range(line[features.index('weight')]):
                    weighted_data.append(line)
            else:
                data[itr] = line
                itr += 1
        if weights and 'weight' in features:
            data = weighted_data
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
        self.__filtered_data = list(data)
        self.x_features = features[:-1]
        self.y_feature = None
        self.__max_min = []
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
        self.add_exclusive_filter(self.y_feature,'=','FLAG')
        self.overwrite_data_w_filtered_data()
        return True

    #append data within self.data to self.filtered_data satisfying (operator,threshold) for a feature
    def add_inclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        index = self.__features.index(feature)
        if self.__filtered_data == self.__data: filtered_data = []
        else: filtered_data = self.__filtered_data
        for line in self.__data:
            if line in filtered_data: continue
            if line[index] is None:
                continue
            elif type(line[index]) == str:
                if operator == "<>":
                    if not (line[index]) == threshold:
                        filtered_data.append(line)
                elif operator == "=":
                    if (line[index]) == threshold:
                        filtered_data.append(line)
            elif line[index] > threshold and '>' in operator:
                filtered_data.append(line)
            elif line[index] == threshold and '=' in operator:
                filtered_data.append(line)
            elif line[index] < threshold and '<' in operator:
                filtered_data.append(line)
            elif str(threshold) in str(line[index]) and 'contains' in operator:
                filtered_data.append(line)
        self.__filtered_data = filtered_data
        return True

    #remove data from self.filtered_data not satisfying (operator,threshold) for a feature
    def add_exclusive_filter(self, feature, operator, threshold):
        if feature not in self.__features:
            print("can't find [{}] in features".format(feature))
            return False
        index = self.__features.index(feature)
        filtered_data = list(self.__filtered_data)
        remove_list = []
        for line in filtered_data:
            if line[index] is None:
                continue
            elif type(line[index]) == str:
                if operator == "<>":
                    if not (line[index]) == threshold:
                        remove_list.append(line)
                elif operator == "=":
                    if (line[index]) == threshold:
                        remove_list.append(line)
            elif  '>' in operator and line[index] > threshold:
                remove_list.append(line)
                #filtered_data.remove(line)
            elif '=' in operator and line[index] == threshold:
                remove_list.append(line)
                #filtered_data.remove(line)
            elif '<' in operator and line[index] < threshold:
                remove_list.append(line)
                #filtered_data.remove(line)
            elif 'contains' in operator and str(threshold) in str(line[index]):
                remove_list.append(line)
                #filtered_data.remove(line)
        for line in remove_list:
            filtered_data.remove(line)
        self.__filtered_data = filtered_data
        return True

    def remove_all_filters(self):
        self.__filtered_data = list(self.__data)
        return True

    #the equivalent of parsing a CSV of filtered data
    def overwrite_data_w_filtered_data(self):
        self.__data = self.__filtered_data

    #todo clarify, different from 's' normalization?
    def std_normalization(self, features = None):
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
            x = [i[0] for i in x]
            result = []
            for i in x:
                if i is not None:
                    result.append(i)
            avg = sum(result)/len(result)
            sqr_err = 0
            for i in result:
                sqr_err += (i-avg)**2
            sqr_err /= len(result)
            std_err = sqr_err**0.5
            result = []
            for i in x:
                if i is None:
                    result.append(None)
                else:
                    result.append((i-avg)/std_err)
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
            if self.__max_min[0][self.__features.index(feature)] is None:
                print("Feature [{}] is not numerical".format(feature))
                return False
        if normalization_type == 's':
            for feature in features:
                result = []
                index = self.__features.index(feature)
                cur_max = self.__max_min[0][index]
                cur_min = self.__max_min[1][index]
                if cur_max is None:
                    print ('feature[{}] max is none'.format(feature))
                elif cur_min is None:
                    print ('feature[{}] min is none'.format(feature))
                for line in self.__data:
                    if line[index] is None:
                        result.append(None)
                    else:
                        result.append((line[index]-cur_min)/(cur_max-cur_min))
                self.add_feature('N_{}'.format(feature), result)
                    #line[index] = (line[index]-cur_min)/(cur_max-cur_min)
                #for line in self.__filtered_data:
                #    line[index] = (line[index]-cur_min)/(cur_max-cur_min)
        elif normalization_type == 't':
            all_max = max([self.__max_min[0][x] for x in range(len(self.__features)) if self.__features[x] in features])
            all_min = min([self.__max_min[1][x] for x in range(len(self.__features)) if self.__features[x] in features])
            for feature in features:
                result = []
                index = self.__features.index(feature)
                self.__max_min[0][index] = all_max
                self.__max_min[1][index] = all_min
                for line in self.__data:
                    if line[index] is None:
                        result.append(None)
                    else:
                        result.append((line[index]-all_min)/(all_max-all_min))
                self.add_feature('N_{}'.format(feature), result)
                    #line[index] = (line[index]-all_min)/(all_max-all_min)
                #for line in self.__filtered_data:
                #    line[index] = (line[index]-all_min)/(all_max-all_min)
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
        if isinstance(features, str):
            features = [features]
        for feature in features:
            if feature not in self.__features:
                print("can't find [{}] in features".format(feature))
                return None
        output = []
        index_list = [self.__features.index(feature) for feature in features]

        for line in self.__filtered_data:
            output.append([line[i] for i in index_list])
        return output

    def get_x_data(self):
        assert (self.y_feature != None), "Must set y feature first before getting x"
        return self.get_data(features=self.x_features)

    def get_y_data(self):
         return self.get_data(features=self.y_feature)

    def __calculate_data_range(self):
        self.__max_min = []
        maxes = list(self.__data[0])
        mins = list(self.__data[0])
        for line in self.__data:
            for i in range(len(line)):
                if maxes[i] is None:
                    maxes[i] = line[i]
                if mins[i] is None:
                    mins[i] = line[i]
                if isinstance(line[i], str):
                    continue
                if line[i] is not None and maxes[i] is not None and line[i] > maxes[i]:
                    maxes[i] = line[i]
                elif line[i] is not None and maxes[i] is not None and line[i] < mins[i]:
                    mins[i] = line[i]
        for i in range(len(self.__features)):
            if isinstance(maxes[i], str):
                maxes[i] = None
                mins[i] = None
        self.__max_min.append(maxes)
        self.__max_min.append(mins)

    def output(self, filename, features=None, data='a'):
        if features is None:
            features = self.__features
        else:
            if isinstance(features, str):
                features = [features]
            for feature in features:
                if feature not in self.__features:
                    print("can't find [{}] in features, no file is created".format(feature))
                    return False
        if data == 'a':
            data = self.__data
        elif data == 'f':
            data = self.__filtered_data
        else:
            print("can't recognize data [{}], please pass in 'a' for all data, or 'f' for filtered data".format(data))
            return False
        f = open(filename, 'w')
        index_list = [self.__features.index(feature) for feature in features]
        for i in range(len(index_list)):
            if i != len(index_list)-1:
                f.write('{},'.format(features[i]))
            else:
                f.write('{}\n'.format(features[i]))
        for line in data:
            for i in range(len(index_list)):
                if i != len(index_list)-1:
                    f.write('{},'.format(line[index_list[i]]))
                else:
                    f.write('{}\n'.format(line[index_list[i]]))
        f.close()
        return True

    def add_feature(self, name, value):
        if len(value) != len(self.__data):
            print ('unmatch data length')
            return False
        self.__features.append(name)
        for i in range(len(value)):
            if value[i] is None:
                self.__data[i].append('')
                continue
            self.__data[i].append(value[i])
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
        index = self.__features.index(feature)
        result = list()
        for line in self.__data:
            if line[index] is None:
                result.append(None)
            else:
                result.append((line[index]-gmin)/(gmax-gmin))
        self.add_feature(newname, result)
        return True
