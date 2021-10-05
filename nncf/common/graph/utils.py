"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import List


def get_concat_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns concatenation axis by given input and output shape of concat node.

    :param input_shapes: Input_shapes of given concat node.
    :param output_shapes: Input_shapes of given concat node.
    :returns: Concatenation axis of given concat node.
    """
    axis = None
    # If it's dummy concat of one tensor
    if len(input_shapes) == 1:
        axis = -1
    else:
        none_dim = None
        for idx, (dim_in, dim_out) in enumerate(zip(input_shapes[0], output_shapes[0])):
            if dim_in != dim_out:
                axis = idx
                break
            elif dim_in is None:
                none_dim = idx
        if not axis:
            axis = none_dim

    if axis is None:
        raise RuntimeError('Unexpected behaviour for concat op')

    return axis
