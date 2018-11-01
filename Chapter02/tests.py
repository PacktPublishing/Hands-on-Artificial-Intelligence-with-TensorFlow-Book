import string
import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion

TF_VERSION_MIN = '1.1'


def check_tf(tf_):
    error_message = "Please use TensorFlow {} or later".format(TF_VERSION_MIN)
    tf_version = tf_.__version__
    condition = LooseVersion(tf_version) > LooseVersion(TF_VERSION_MIN)
    assert condition, error_message
    _log('Using TensorFlow {}'.format(tf_version))


def test_case(f):
    def run(*args, **kwargs):
        f(*args, **kwargs)
        _success_message()
    return run


def _log(message):
    print(message)


def _success_message():
    _log('✓ Tests passed')


def _run_test(f, actuals, expected, messages):
    for actual, exp, msg in zip(actuals, expected, messages):
        assert f(*actual) == exp, msg


def _evaluate_tensor(tensor):
    return tf.Session().run(tensor)


def _get_random_np_array():
    no_rows, no_cols = np.random.randint(1, 5, 2)
    value = np.random.randint(1, 100)
    return np.full([no_rows, no_cols], fill_value=value)


def _get_random_string(string_length=10):
    return ''.join(np.random.choice(list(string.ascii_lowercase),
                                    string_length))


# Basic tensor operations
@test_case
def test_create_tensor_from_list(f):
    np_array_expected = _get_random_np_array()
    list_expected = np_array_expected.tolist()
    tensor = f(list_expected)
    np_array_actual = _evaluate_tensor(tensor)
    assert list_expected == np_array_actual.tolist()


@test_case
def test_create_tensor_from_np_array(f):
    np_array_expected = _get_random_np_array()
    tensor_name = _get_random_string()
    tensor = f(np_array_expected, name=tensor_name)
    numpy_array_actual = _evaluate_tensor(tensor)
    np.array_equal(np_array_expected, numpy_array_actual)


@test_case
def test_get_tensor_name(f):
    name_expected = _get_random_string()
    tensor = tf.constant(value=0, name=name_expected)
    name_actual = f(tensor)
    assert '{}:0'.format(name_expected) == name_actual


@test_case
def test_get_tensor_shape(f):
    shape = [6, 2]
    tensor_shape = f(tf.constant(0, shape=shape))
    assert tensor_shape == shape


@test_case
def test_get_tensor_rank(f):
    shape = [3, 7]
    rank = f(tf.constant(0, shape=shape))
    assert rank == len(shape)


@test_case
def test_get_tensor_dtype(f):
    shape = [3, 7]
    value = 1.2
    dtype = f(tf.constant(value, shape=shape))
    assert dtype == tf.float32
    return _success_message()


@test_case
def test_create_constant_tensor(f):
    value = 42
    m = 5
    n = 3
    tensor = f(value, m, n)
    array_tf = _evaluate_tensor(tensor)
    array_np = np.full(shape=[m, n], fill_value=value)
    np.array_equal(array_tf, array_np)


@test_case
def test_create_fill_tensor(f):
    value = np.random.randint(1, 10)
    m = np.random.randint(1, 10)
    n = np.random.randint(1, 10)
    tensor = f(value, m, n)
    array_tf = _evaluate_tensor(tensor)
    array_np = np.full(shape=[m, n], fill_value=value)
    np.array_equal(array_tf, array_np)


# Using scopes
@test_case
def test_create_variable_in_scope(f):
    name = _get_random_string()
    scope_name = _get_random_string()
    np_array = _get_random_np_array()
    tensor = f(name, np_array, scope_name)
    name_expected = '{}/{}:0'.format(scope_name, name)
    assert tensor.name == name_expected


@test_case
def test_create_variable_in_nested_scope(f):
    name = _get_random_string()
    scope_name_outer = _get_random_string()
    scope_name_inner = _get_random_string()
    np_array = _get_random_np_array()
    tensor = f(name, np_array, scope_name_outer, scope_name_inner)
    name_expected = '{}/{}/{}:0'.format(scope_name_outer,
                                        scope_name_inner,
                                        name)
    assert tensor.name == name_expected


# Using multiple graphs
@test_case
def test_get_default_graph(f):
    default_graph = f()
    assert default_graph is tf.get_default_graph()


@test_case
def test_create_new_graph(f):
    g = f()
    assert g is not tf.get_default_graph()


@test_case
def test_get_graph_seed(f):
    seed = np.random.randint(0, 1000)
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(seed)
    assert f(g) == seed


@test_case
def test_set_graph_seed(f):
    seed = np.random.randint(0, 1000)
    g = tf.Graph()
    graph_actual = f(g, seed)
    assert graph_actual.seed == seed


# Math operations
@test_case
def test_add(f):
    actuals = [(1, 0), (2, 1)]
    expected = [1, 3]
    messages = ['add(1, 0) should return 1', 'add(2, 1) should return 3']
    _run_test(f, actuals, expected, messages)


@test_case
def test_add_rank0_tensors(f):
    x = 1
    y = 2
    z_tensor = f(x, y)
    z = tf.Session().run(z_tensor)
    assert z == x + y


@test_case
def test_add_rank1_tensors(f):
    xs = [1, 2, 3]
    ys = [6, 5, 4]
    z_tensor = f(xs, ys)
    z = tf.Session().run(z_tensor)
    assert np.all(z == np.array([x + y for x, y in zip(xs, ys)]))
