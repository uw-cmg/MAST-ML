import standardized.parser as parser
import pytest
import os

__author__ = 'haotian'


test_file_name = 'temp_testdata.csv'
test_file_content = [['col1', 'col2', 'col3', 'col4'],
                        ['row1', 1, 2.1, 4],
                        ['row2', 2, 3.4, 8],
                        ['row3', 3, 4.5, 12]]


@pytest.fixture(scope='session', autouse=True)  # this function is excuted before all tests
def create_testfile(request):
    f = open(test_file_name, 'w')
    for row in test_file_content:
        f.write('{}\n'.format(",".join([str(x) for x in row])))
    f.close()

    def delete_testfile():
        os.remove(test_file_name)
    request.addfinalizer(delete_testfile)  # delete the temp test csv after all tests are done


# this function is excuted everytime a test that is called and this function is added as the parameter
@pytest.fixture(scope='function')
def test_data():
    return parser.parse(test_file_name)


def test_parse_invalid_filename():
    data = parser.parse('no file')
    assert data is None, "invalid file name, data object should be none"


def test_parse_valid_filename():
    data = parser.parse(test_file_name)
    assert data is not None, "valid file name, data object should not be none"


def test_to_digit_with_int_string():
    x = parser.to_digit('3')
    assert isinstance(x, int), "\'3\' should be converted to int"
    assert x == 3, "\'3\' should be converted to 3"


def test_to_digit_with_float_string():
    x = parser.to_digit('3.4')
    assert isinstance(x, float), "\'3.4\' should be converted to float"
    assert x == 3.4, "\'3.4\' should be converted to 3.4"


def test_to_digit_with_non_digit_string():
    x = parser.to_digit('abc')
    assert isinstance(x, str), "\'abc'\ should be converted to float"
    assert x == 'abc', "\'abc\' should be converted to \'abc\'"


def test_get_data_return_content(test_data):
    assert test_data.get_data() == test_file_content[1:]