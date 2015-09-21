import collections
from itertools import izip
import functools
from math import sin, cos, radians


class DimensionError(TypeError):
    """
    Raised by the requires_same_dimensions decorator when the
    component Vectors are not of the same dimension
    """
    pass


def requires_same_dimensions(func):
    """
    Decorator for methods that operate on tensors which require the
    tensors to be of the same dimension

    Assumes the first two arguments to the decorated function will be a
    Vector or Matrix followed by another Vector, Matrix, or sequence.
    """
    @functools.wraps(func)
    def decorated_func(l_operand, r_operand, *args, **kwargs):
        try:
            same_dimension = len(l_operand) != len(r_operand)
        except TypeError:   # no len
            raise TypeError('Operation requires sequence, not '
                            "'%s'." % r_operand.__class__.__name__)
        if same_dimension:
            err = ('Operation requires %i-element %s or sequence, '
                   '%i-element %s given.' % (len(l_operand),
                                             l_operand.__class__.__name__,
                                             len(r_operand),
                                             r_operand.__class__.__name__))
            raise DimensionError(err)
        return func(l_operand, r_operand, *args, **kwargs)
    return decorated_func


class TensorIterator(collections.Iterator):
    """Iterator for the Vector and Matrix classes"""
    def __init__(self, tensor):
        self.index = -1
        self.tensor = tensor

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        try:
            return self.tensor[self.index]
        except IndexError:
            raise StopIteration


class Tensor(tuple):
    """Base class for Vector and Matrix"""

    def __repr__(self):
        return u'%s(%s)' % (self.__class__.__name__,
                            u', '.join(map(unicode, self)))

    def __iter__(self):
        return TensorIterator(self)

    def __radd__(self, operand):
        return self.__add__(operand)

    def __rmul__(self, operand):
        return self.__mul__(operand)

    def __rsub__(self, operand):
        return self.__sub__(operand)

    # Division is supported when the Tensor is the dividend,
    # and is interpreted as multiplying by the reciprocal of
    # the divisor.
    def __div__(self, operand):
        return self.__mul__(1.0 / operand)

    def __truediv__(self, operand):
        return self.__mul__(1.0 / operand)

    # The unary `-` operator is interpreted as scalar
    # multiplication by -1. `+` is supported but a no-op.
    def __neg__(self):
        return self.__class__(-elem for elem in self)

    def __pos__(self):
        return self


class Vector(Tensor):
    """
    An immutable primitive class for storing vectors of arbitrary
    dimension as a special type of tuple.
    """
    def __new__(cls, *args):
        if len(args) > 1:
            seq = tuple(map(float, args))
        else:
            if isinstance(args[0], Matrix):
                seq = tuple(map(float, args[0][0]))
            else:
                seq = tuple(map(float, args[0]))

        self = super(Vector, cls).__new__(cls, tuple(seq))
        try:
            self.x, self.y, self.z = seq[:3]
        except ValueError:
            try:
                self.x, self.y = seq
                self.z = 0.0
            except ValueError:
                self.x = seq[0]
                self.y = self.z = 0.0

        self._seq = seq
        self._magnitude_squared = None
        self._magnitude = None
        return self

    # Vector addition and subtraction is supported
    # for vectors of the same size
    @requires_same_dimensions
    def __add__(self, operand):
        return Vector(self._seq[i] + elem for i, elem in enumerate(operand))

    @requires_same_dimensions
    def __sub__(self, operand):
        return Vector(self._seq[i] - elem for i, elem in enumerate(operand))

    # Multiplication is interpreted as scalar multiplication
    # and tries to coerce the multiplier to a float
    def __mul__(self, operand):
        try:
            scalar = float(operand)
        except TypeError:
            raise TypeError('Can only multiply Vector by scalar, '
                            'not %s.' % operand.__class__.__name__)
        return Vector(elem * scalar for elem in self)

    def cross(self, operand):
        """
        Compute the cross product of this Vector and another
        Vector or sequence

        Because this operation is only defined in 3 dimensions,
        both operands are coerced to 3-dimensional vectors (either
        by substituting 0 for missing values or ignoring all but
        the first three entries)
        """
        vector = Vector(operand)
        return Vector(self.y * vector.z - self.z * vector.y,
                      self.z * vector.x - self.x * vector.z,
                      self.x * vector.y - self.y * vector.x)

    @requires_same_dimensions
    def dot(self, vector):
        """
        Compute the dot product of this vector and another
        Vector or sequence
        """
        return sum(a*b for a, b in izip(self, vector))

    @property
    def magnitude_squared(self):
        """
        The square of the magnitude of the vector (the sum of the
        squares of its elements). This value is lazily evaluated
        once and then cached.
        """
        if self._magnitude_squared is None:
            self._magnitude_squared = sum(elem**2 for elem in self)
        return self._magnitude_squared

    @property
    def magnitude(self):
        """
        The magnitude (norm) of the vector. This value is lazily
        evaluated once and then cached.
        """
        if self._magnitude is None:
            self._magnitude = self.magnitude_squared ** 0.5
        return self._magnitude


class Matrix(Tensor):
    """
    An immutable primitive class for storing Matrices of arbitrary
    dimension as a collection of Vectors.
    """
    def __new__(cls, *args):
        @requires_same_dimensions
        def _add_col(v1, v2):
            """
            Check each column is the same size as the first as the Matrix
            is constructed, raising DimensionError otherwise
            """
            cols.append(v2)

        if len(args) != 1:
            sequence_list = iter(args)
        elif isinstance(args[0], Vector):
            sequence_list = iter((args[0],))
        else:
            sequence_list = iter(args[0])

        reference_column = Vector(next(sequence_list))
        cols = [reference_column]
        try:
            while True:
                _add_col(reference_column, Vector(next(sequence_list)))
        except StopIteration:
            pass

        cols = tuple(cols)
        self = super(Matrix, cls).__new__(cls, cols)
        self.cols = cols
        self._rows = None
        return self

    # Matrix addition and subtraction implicitly relies on vector
    # addition for the underlying columns
    @requires_same_dimensions
    def __add__(self, operand):
        return Matrix(col + Vector(operand[i]) for i, col in enumerate(self))

    @requires_same_dimensions
    def __sub__(self, operand):
        return Matrix(col - Vector(operand[i]) for i, col in enumerate(self))

    # Unlike with Vector, Matrix tries scalar multiplication but
    # "falls back" to matrix mutiplication in the event the multiplier
    # cannot be ocerced to float
    def __mul__(self, operand):
        try:
            scalar = float(operand)
        except TypeError:
            # Switch to Matrix Multiplication
            operand = Matrix(operand)
            if len(self.cols) != len(operand.rows):
                raise ValueError('Matrix multiplication is only defined when '
                                 'the number of columns in the left matrix '
                                 'matches the number of rows in the other.')
            result_matrix = []
            for col in operand.cols:
                result_col = []
                for row in self.rows:
                    result_col.append(row.dot(col))
                result_matrix.append(result_col)
            return Matrix(result_matrix)
        else:
            return Matrix(scalar * col for col in self)

    @classmethod
    def identity(cls, n):
        """Construct and return an n x n Identity Matrix"""
        return cls([(1 if j == i else 0) for j in range(n)] for i in range(n))

    @staticmethod
    def rotate(rotations):
        """
        Construct and return a rotation Matrix as described by the
        parameter rotations.

        Rotations should be a sequence of pairs where the first element
        is a string describing the axis of rotation ('x', 'y', or 'z')
        and the second is a float describing the counterclockwise
        rotation angle. The rotation angle should be in radians, but
        if a high rotation angle is given it will be assumed to be degrees
        and converted.

        Example usage:
        # matrix for rotating first 90 degrees around the z-axis,
        # followed by 90 degrees around the x-axis.
        Matrix.rotate([('z', 90), ('x', 90)])
        # alternatively, for the same matrix
        Matrix.rotate([('z', math.pi/2), ('x', math.pi/2)])
        """
        rotation = 1
        for axis, angle in rotations:
            if angle > 7:   # too high so probably in degrees
                angle = radians(angle)

            if axis == 'x':
                rotation = Matrix((1, 0, 0),
                                  (0, cos(angle), sin(angle)),
                                  (0, -sin(angle), cos(angle))) * rotation
            elif axis == 'y':
                rotation = Matrix((cos(angle), 0, sin(angle)),
                                  (0, 1, 0),
                                  (-sin(angle), 0, cos(angle))) * rotation
            elif axis == 'z':
                rotation = Matrix((cos(angle), sin(angle), 0),
                                  (-sin(angle), cos(angle), 0),
                                  (0, 0, 1)) * rotation

        return rotation

    @property
    def rows(self):
        """Return the rows of this Matrix as a tuple of Vectors"""
        if self._rows is None:
            self._rows = tuple(Vector(seq) for seq in izip(*self.cols))
        return self._rows

    def transpose(self):
        """Return a new Matrix representing the transpose of this one"""
        return Matrix(self.rows)


class Quaternion(Vector):
    def __new__(cls, s, *args):
        i, j, k = args if len(args) != 1 else args[0]
        self = super(Quaternion, cls).__new__(cls, s, i, j, k)
        self.s = self[0]
        self.v = self.x, self.y, self.z = Vector(self[1:])
        return self

    def __repr__(self):
        return 'Quaternion(%s, %s)' % (self.s, self.v)

    def __mul__(self, operand):
        try:
            scaled_vector = super(Quaternion, self).__mul__(operand)
        except TypeError:
            # Switch to Quaternion multiplication
            b = Quaternion(*operand)
            s1, s2 = self.s, b.s
            v1, v2 = self.v, b.v
            return Quaternion(s1*s2 - v1.dot(v2), v1.cross(v2) + s1*v2 + s2*v1)
        else:
            return Quaternion(*scaled_vector)

    def __rdiv__(self, operand):
        return operand * self.inverse()

    def __rtruediv__(self, operand):
        return operand * self.inverse()

    def conjugate(self):
        """Return the conjugate of this Quaternion as a new Quaternion"""
        return Quaternion(self.s, (-self.x, -self.y, -self.z))

    def inverse(self):
        """Return the inverse of this Quaternion as a Quaternion"""
        return self.conjugate() / self.magnitude_squared

    @property
    def scalar(self):
        """Return the scalar part of the Quaternion"""
        return self.s

    @property
    def vector(self):
        """Return the Vector part of the Quaternion, as a Vector"""
        return self.v
