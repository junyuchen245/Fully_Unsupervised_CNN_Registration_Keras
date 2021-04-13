"""
Image warping using per-pixel flow vectors based on tensorflow's _interpolate_bilinear

Added support for:
1. 2D/3D nearest-neighbor & bilinear interpolation.
2. 2D/3D image warping

Author:
  Junyu Chen
Email:
  jchen245@jhmi.edu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
  """Similar to Matlab's interp2 function.
  Finds values for query points on a grid using bilinear interpolation.
  Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).
  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`
  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, channels = (array_ops.shape(grid)[0],
                                           array_ops.shape(grid)[1],
                                           array_ops.shape(grid)[2],
                                           array_ops.shape(grid)[3])

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.'),
        check_ops.assert_equal(
            array_ops.shape(query_points)[2],
            2,
            message='Query points must be size 2 in dim 2.')
    ]):
      num_queries = array_ops.shape(query_points)[1]

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            height, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1] if indexing == 'ij' else [1, 0]
      unstacked_query_points = array_ops.unstack(query_points, axis=2)

    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    with ops.control_dependencies([
        check_ops.assert_less_equal(
            math_ops.cast(batch_size * height * width, dtype=dtypes.float32),
            np.iinfo(np.int32).max / 8,
            message="""The image size or batch size is sufficiently large
                       that the linearized addresses used by array_ops.gather
                       may exceed the int32 limit.""")
    ]):
      flattened_grid = array_ops.reshape(
          grid, [batch_size * height * width, channels])
      batch_offsets = array_ops.reshape(
          math_ops.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with ops.name_scope('interpolate'):
      interp_top = alphas[1] * (top_right - top_left) + top_left
      interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
      interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

def _interpolate_bilinear3D(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
  """
  3D bilinear interpolation based on tensorflow's _interpolate_bilinear
  Author:
    Junyu Chen
  Email:
    jchen245@jhmi.edu
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 5:
      msg = 'Grid must be 5 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, length, channels = (array_ops.shape(grid)[0],
                                                array_ops.shape(grid)[1],
                                                array_ops.shape(grid)[2],
                                                array_ops.shape(grid)[3],
                                                array_ops.shape(grid)[4])

    shape = [batch_size, length, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.'),
        check_ops.assert_equal(
            array_ops.shape(query_points)[2],
            3,
            message='Query points must be size 2 in dim 2.')
    ]):
      num_queries = array_ops.shape(query_points)[1]

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            height, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1, 2] if indexing == 'ij' else [2, 1, 0]

      unstacked_query_points = array_ops.unstack(query_points, axis=2)
    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    with ops.control_dependencies([
        check_ops.assert_less_equal(
            math_ops.cast(batch_size * height * width * length, dtype=dtypes.float32),
            np.iinfo(np.int32).max / 8,
            message="""The image size or batch size is sufficiently large
                       that the linearized addresses used by array_ops.gather
                       may exceed the int32 limit.""")
    ]):
      flattened_grid = array_ops.reshape(
          grid, [batch_size * height * width * length, channels])
      batch_offsets = array_ops.reshape(
          math_ops.range(batch_size) * height * width * length, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(z_coords, x_coords, y_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + (y_coords*width+z_coords)*length + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 8 corners around each query point
    btop_left = gather(floors[2], floors[0], floors[1], 'btop_left')
    btop_right = gather(floors[2], floors[0], ceils[1], 'btop_right')
    bbottom_left = gather(floors[2], ceils[0], floors[1], 'bbottom_left')
    bbottom_right = gather(floors[2], ceils[0], ceils[1], 'bbottom_right')

    ttop_left = gather(ceils[2], floors[0], floors[1], 'ttop_left')
    ttop_right = gather(ceils[2], floors[0], ceils[1], 'ttop_right')
    tbottom_left = gather(ceils[2], ceils[0], floors[1], 'tbottom_left')
    tbottom_right = gather(ceils[2], ceils[0], ceils[1], 'tbottom_right')

    # now, do the actual interpolation
    #alphas = [0.1,0.1,0.1]

    with ops.name_scope('interpolate'):
      interp_top = alphas[1] * (btop_right - btop_left) + btop_left
      interp_bottom = alphas[1] * (bbottom_right - bbottom_left) + bbottom_left
      interp1 = alphas[0] * (interp_bottom - interp_top) + interp_top

      interp_top = alphas[1] * (ttop_right - ttop_left) + ttop_left
      interp_bottom = alphas[1] * (tbottom_right - tbottom_left) + tbottom_left
      interp2 = alphas[0] * (interp_bottom - interp_top) + interp_top

      interp = alphas[2] * (interp2 - interp1) + interp1

    return interp#math_ops.cast(floors[2], dtype=grid_type)

def _interpolate_nearest(grid,
                          query_points,
                          name='interpolate_nearest',
                          indexing='ij'):
  """
  2D nearest-neighbor interpolation based on tensorflow's _interpolate_bilinear
  Author:
    Junyu Chen
  Email:
    jchen245@jhmi.edu
  """

  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, channels = (array_ops.shape(grid)[0],
                                           array_ops.shape(grid)[1],
                                           array_ops.shape(grid)[2],
                                           array_ops.shape(grid)[3])

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.'),
        check_ops.assert_equal(
            array_ops.shape(query_points)[2],
            2,
            message='Query points must be size 2 in dim 2.')
    ]):
      num_queries = array_ops.shape(query_points)[1]

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            height, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1] if indexing == 'ij' else [1, 0]
      unstacked_query_points = array_ops.unstack(query_points, axis=2)

    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    with ops.control_dependencies([
        check_ops.assert_less_equal(
            math_ops.cast(batch_size * height * width, dtype=dtypes.float32),
            np.iinfo(np.int32).max / 8,
            message="""The image size or batch size is sufficiently large
                       that the linearized addresses used by array_ops.gather
                       may exceed the int32 limit.""")
    ]):
      flattened_grid = array_ops.reshape(
          grid, [batch_size * height * width, channels])
      batch_offsets = array_ops.reshape(
          math_ops.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with ops.name_scope('interpolate'):
      interp_top = K.round(alphas[1]) * (top_right - top_left) + top_left
      interp_bottom = K.round(alphas[1]) * (bottom_right - bottom_left) + bottom_left
      interp = K.round(alphas[0]) * (interp_bottom - interp_top) + interp_top
      # Interpolate value to labels with higher intensity values
      max1 = K.maximum(top_left, top_right)
      max2 = K.maximum(bottom_left, bottom_right)
      interp = K.maximum(max1, max2)
    return interp

def _interpolate_nearest3D(grid,
                          query_points,
                          name='interpolate_nearest3D',
                          indexing='ij'):
  """
  3D nearest-neighbor interpolation based on tensorflow's _interpolate_bilinear
  Author:
    Junyu Chen
  Email:
    jchen245@jhmi.edu
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 5:
      msg = 'Grid must be 5 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, length, channels = (array_ops.shape(grid)[0],
                                                array_ops.shape(grid)[1],
                                                array_ops.shape(grid)[2],
                                                array_ops.shape(grid)[3],
                                                array_ops.shape(grid)[4])

    shape = [batch_size, length, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.'),
        check_ops.assert_equal(
            array_ops.shape(query_points)[2],
            3,
            message='Query points must be size 2 in dim 2.')
    ]):
      num_queries = array_ops.shape(query_points)[1]

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            height, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1, 2] if indexing == 'ij' else [2, 1, 0]

      unstacked_query_points = array_ops.unstack(query_points, axis=2)
    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    with ops.control_dependencies([
        check_ops.assert_less_equal(
            math_ops.cast(batch_size * height * width * length, dtype=dtypes.float32),
            np.iinfo(np.int32).max / 8,
            message="""The image size or batch size is sufficiently large
                       that the linearized addresses used by array_ops.gather
                       may exceed the int32 limit.""")
    ]):
      flattened_grid = array_ops.reshape(
          grid, [batch_size * height * width * length, channels])
      batch_offsets = array_ops.reshape(
          math_ops.range(batch_size) * height * width * length, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(z_coords, x_coords, y_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + (y_coords*width+z_coords)*length + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 8 corners around each query point
    btop_left = gather(floors[2], floors[0], floors[1], 'btop_left')
    btop_right = gather(floors[2], floors[0], ceils[1], 'btop_right')
    bbottom_left = gather(floors[2], ceils[0], floors[1], 'bbottom_left')
    bbottom_right = gather(floors[2], ceils[0], ceils[1], 'bbottom_right')

    ttop_left = gather(ceils[2], floors[0], floors[1], 'ttop_left')
    ttop_right = gather(ceils[2], floors[0], ceils[1], 'ttop_right')
    tbottom_left = gather(ceils[2], ceils[0], floors[1], 'tbottom_left')
    tbottom_right = gather(ceils[2], ceils[0], ceils[1], 'tbottom_right')

    # now, do the actual interpolation
    #alphas = [0.1,0.1,0.1]

    with ops.name_scope('interpolate'):
      interp_top = K.round(alphas[1]) * (btop_right - btop_left) + btop_left
      interp_bottom = K.round(alphas[1]) * (bbottom_right - bbottom_left) + bbottom_left
      interp1 = K.round(alphas[0]) * (interp_bottom - interp_top) + interp_top

      interp_top = K.round(alphas[1]) * (ttop_right - ttop_left) + ttop_left
      interp_bottom = K.round(alphas[1]) * (tbottom_right - tbottom_left) + tbottom_left
      interp2 = K.round(alphas[0]) * (interp_bottom - interp_top) + interp_top

      interp = K.round(alphas[2]) * (interp2 - interp1) + interp1
    return interp

def image_warp(image, flow, interp_method, name='dense_image_warp'):
  """Image warping using per-pixel flow vectors.
  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by bilinear
  interpolation of the 4 nearest pixels around
  (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.
  Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).
    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.
  Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.
  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  """
  with ops.name_scope(name):
    batch_size, height, width, channels = (array_ops.shape(image)[0],
                                           array_ops.shape(image)[1],
                                           array_ops.shape(image)[2],
                                           array_ops.shape(image)[3])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = array_ops.meshgrid(
        math_ops.range(width), math_ops.range(height))
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    # plus or minus
    # plus for vecInt

    query_points_on_grid = batched_grid - flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    if interp_method == 'bilinear' or interp_method == 'Bilinear':
        interpolated = _interpolate_bilinear(image, query_points_flattened)
    elif interp_method == 'nearest_neighbor' or interp_method == 'NN':
        interpolated = _interpolate_nearest(image, query_points_flattened)
    else:
        print('Running on bi-linear interpolation!')
        interpolated = _interpolate_bilinear(image, query_points_flattened)
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return interpolated

def image_warp3D(image, flow, interp_method, name='dense_image_warp'):
  with ops.name_scope(name):
    batch_size, height, width, length, channels = (array_ops.shape(image)[0],
                                                array_ops.shape(image)[1],
                                                array_ops.shape(image)[2],
                                                array_ops.shape(image)[3],
                                                array_ops.shape(image)[4])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y, grid_z = array_ops.meshgrid(
        math_ops.range(height), math_ops.range(width), math_ops.range(length))
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_z, grid_y, grid_x], axis=3), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    print(image.get_shape().as_list())
    #
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width * length, 3])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    if interp_method == 'bilinear' or interp_method == 'Bilinear':
        interpolated = _interpolate_bilinear3D(image, query_points_flattened)
    elif interp_method == 'nearest_neighbor' or interp_method == 'NN':
        interpolated = _interpolate_nearest3D(image, query_points_flattened)
    else:
        print('Running on bi-linear interpolation!')
        interpolated = _interpolate_bilinear3D(image, query_points_flattened)
    interpolated = array_ops.reshape(interpolated, [batch_size, height, width, length, channels])
    return interpolated


def Mapping2D_nn(input):
    a = input[0]
    b = input[1]
    output = image_warp(a,b,'NN',name='dense_image_warp')
    return output

def Mapping3D_nn(input):
    a = input[0]
    b = input[1]
    output = image_warp3D(a, b, 'NN', name='dense_image_warp')
    return output

def Mapping2D(input):
    a = input[0]
    b = input[1]
    output = image_warp(a, b, 'bilinear', name='dense_image_warp')
    return output

def Mapping3D(input):
    a = input[0]
    b = input[1]
    output = image_warp3D(a,b, 'bilinear',name='dense_image_warp')
    return output