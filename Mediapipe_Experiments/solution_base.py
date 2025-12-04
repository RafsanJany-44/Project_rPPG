# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone SolutionBase using local MediaPipe bindings and protos."""

import collections
import enum
import os
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import numpy as np

from google.protobuf.internal import containers
from google.protobuf import descriptor
from google.protobuf import message

# ---- Local protobuf and helper modules (copied into this folder) ----
import calculator_pb2
import classification_pb2
import detection_pb2
import landmark_pb2
import rect_pb2

import packet_creator
import packet_getter

# ---- Local C++ bindings (from _framework_bindings.cp38-win_amd64.pyd) ----
from _framework_bindings import calculator_graph
from _framework_bindings import image_frame
from _framework_bindings import packet
from _framework_bindings import resource_util
from _framework_bindings import validated_graph_config


RGB_CHANNELS = 3

# We don't support modifying calculator options in this minimal setup.
CALCULATOR_TO_OPTIONS: Mapping[str, Any] = {}


def type_names_from_oneof(oneof_type_name: str) -> Optional[List[str]]:
  if oneof_type_name.startswith('OneOf<') and oneof_type_name.endswith('>'):
    comma_separated_types = oneof_type_name[len('OneOf<'):-len('>')]
    return [n.strip() for n in comma_separated_types.split(',')]
  return None


@enum.unique
class PacketDataType(enum.Enum):
  """The packet data types supported by the SolutionBase class."""
  STRING = 'string'
  BOOL = 'bool'
  BOOL_LIST = 'bool_list'
  INT = 'int'
  INT_LIST = 'int_list'
  FLOAT = 'float'
  FLOAT_LIST = 'float_list'
  AUDIO = 'matrix'
  IMAGE = 'image'
  IMAGE_LIST = 'image_list'
  IMAGE_FRAME = 'image_frame'
  PROTO = 'proto'
  PROTO_LIST = 'proto_list'

  @staticmethod
  def from_registered_name(registered_name: str) -> 'PacketDataType':
    try:
      return NAME_TO_TYPE[registered_name]
    except KeyError as e:
      names = type_names_from_oneof(registered_name)
      if names:
        for n in names:
          if n in NAME_TO_TYPE.keys():
            return NAME_TO_TYPE[n]
      raise e


NAME_TO_TYPE: Mapping[str, 'PacketDataType'] = {
    'string':
        PacketDataType.STRING,
    'bool':
        PacketDataType.BOOL,
    '::std::vector<bool>':
        PacketDataType.BOOL_LIST,
    'int':
        PacketDataType.INT,
    '::std::vector<int>':
        PacketDataType.INT_LIST,
    'int64':
        PacketDataType.INT,
    '::std::vector<int64>':
        PacketDataType.INT_LIST,
    'float':
        PacketDataType.FLOAT,
    '::std::vector<float>':
        PacketDataType.FLOAT_LIST,
    '::mediapipe::Matrix':
        PacketDataType.AUDIO,
    '::mediapipe::ImageFrame':
        PacketDataType.IMAGE_FRAME,
    '::mediapipe::Classification':
        PacketDataType.PROTO,
    '::mediapipe::ClassificationList':
        PacketDataType.PROTO,
    '::mediapipe::ClassificationListCollection':
        PacketDataType.PROTO,
    '::mediapipe::Detection':
        PacketDataType.PROTO,
    '::mediapipe::DetectionList':
        PacketDataType.PROTO,
    '::mediapipe::Landmark':
        PacketDataType.PROTO,
    '::mediapipe::LandmarkList':
        PacketDataType.PROTO,
    '::mediapipe::LandmarkListCollection':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmark':
        PacketDataType.PROTO,
    '::mediapipe::FrameAnnotation':
        PacketDataType.PROTO,
    '::mediapipe::Trigger':
        PacketDataType.PROTO,
    '::mediapipe::Rect':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedRect':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkList':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkListCollection':
        PacketDataType.PROTO,
    '::mediapipe::Image':
        PacketDataType.IMAGE,
    '::std::vector<::mediapipe::Image>':
        PacketDataType.IMAGE_LIST,
    '::std::vector<::mediapipe::Classification>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::ClassificationList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Detection>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::DetectionList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Landmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::LandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Rect>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedRect>':
        PacketDataType.PROTO_LIST,
}


class SolutionBase:
  """Common base class for high-level MediaPipe-style solutions."""

  def __init__(
      self,
      binary_graph_path: Optional[str] = None,
      graph_config: Optional[calculator_pb2.CalculatorGraphConfig] = None,
      calculator_params: Optional[Mapping[str, Any]] = None,
      graph_options: Optional[message.Message] = None,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):

    if bool(binary_graph_path) == bool(graph_config):
      raise ValueError(
          "Must provide exactly one of 'binary_graph_path' or 'graph_config'.")

    # In this minimal setup, we treat the directory of this file as the
    # resource root. All .binarypb and .tflite files should live here.
    root_path = os.path.dirname(os.path.abspath(__file__))
    resource_util.set_resource_dir(root_path)

    validated_graph = validated_graph_config.ValidatedGraphConfig()
    if binary_graph_path:
      validated_graph.initialize(
          binary_graph_path=os.path.join(root_path, binary_graph_path))
    else:
      validated_graph.initialize(graph_config=graph_config)

    canonical_graph_config_proto = self._initialize_graph_interface(
        validated_graph, side_inputs, outputs, stream_type_hints)

    # For this minimal version we do not support changing calculator options.
    if calculator_params:
      raise ValueError(
          "calculator_params are not supported in this minimal SolutionBase.")

    if graph_options:
      self._set_extension(canonical_graph_config_proto.graph_options,
                          graph_options)

    self._graph = calculator_graph.CalculatorGraph(
        graph_config=canonical_graph_config_proto)
    self._simulated_timestamp = 0
    self._graph_outputs = {}

    def callback(stream_name: str, output_packet: packet.Packet) -> None:
      self._graph_outputs[stream_name] = output_packet

    for stream_name in self._output_stream_type_info.keys():
      self._graph.observe_output_stream(stream_name, callback, True)

    self._input_side_packets = {
        name: self._make_packet(self._side_input_type_info[name], data)
        for name, data in (side_inputs or {}).items()
    }
    self._graph.start_run(self._input_side_packets)

  def process(
      self, input_data: Union[np.ndarray, Mapping[str, Union[np.ndarray,
                                                             message.Message]]]
  ) -> NamedTuple:
    """Processes RGB image data and returns SolutionOutputs."""
    self._graph_outputs.clear()

    if isinstance(input_data, np.ndarray):
      if len(self._input_stream_type_info.keys()) != 1:
        raise ValueError(
            "Can't process single image input since the graph has more than one input streams."
        )
      input_dict = {next(iter(self._input_stream_type_info)): input_data}
    else:
      input_dict = input_data

    # Simulate ~30 fps video input with a fixed timestamp increment.
    self._simulated_timestamp += 33333
    for stream_name, data in input_dict.items():
      input_stream_type = self._input_stream_type_info[stream_name]
      if (input_stream_type == PacketDataType.PROTO_LIST or
          input_stream_type == PacketDataType.AUDIO):
        raise NotImplementedError(
            f'SolutionBase can only process non-audio and non-proto-list data. '
            f'{self._input_stream_type_info[stream_name].name} '
            f'type is not supported yet.')
      elif (input_stream_type == PacketDataType.IMAGE_FRAME or
            input_stream_type == PacketDataType.IMAGE):
        if data.shape[2] != RGB_CHANNELS:
          raise ValueError('Input image must contain three channel rgb data.')
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))
      else:
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))

    self._graph.wait_until_idle()

    solution_outputs = collections.namedtuple(
        'SolutionOutputs', self._output_stream_type_info.keys())
    for stream_name in self._output_stream_type_info.keys():
      if stream_name in self._graph_outputs:
        setattr(
            solution_outputs, stream_name,
            self._get_packet_content(self._output_stream_type_info[stream_name],
                                     self._graph_outputs[stream_name]))
      else:
        setattr(solution_outputs, stream_name, None)

    return solution_outputs

  def close(self) -> None:
    """Closes all the input sources and the graph."""
    self._graph.close()
    self._graph = None
    self._input_stream_type_info = None
    self._output_stream_type_info = None

  def reset(self) -> None:
    """Resets the graph for another run."""
    if self._graph:
      self._graph.close()
      self._graph.start_run(self._input_side_packets)

  def _initialize_graph_interface(
      self,
      validated_graph: validated_graph_config.ValidatedGraphConfig,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):
    """Gets graph interface type information and returns canonical config proto."""
    canonical_graph_config_proto = calculator_pb2.CalculatorGraphConfig()
    canonical_graph_config_proto.ParseFromString(validated_graph.binary_config)

    def get_name(tag_index_name):
      return tag_index_name.split(':')[-1]

    def get_stream_packet_type(packet_tag_index_name):
      stream_name = get_name(packet_tag_index_name)
      if stream_type_hints and stream_name in stream_type_hints.keys():
        return stream_type_hints[stream_name]
      return PacketDataType.from_registered_name(
          validated_graph.registered_stream_type_name(stream_name))

    self._input_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in canonical_graph_config_proto.input_stream
    }

    if not outputs:
      output_streams = canonical_graph_config_proto.output_stream
    else:
      output_streams = outputs
    self._output_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in output_streams
    }

    def get_side_packet_type(packet_tag_index_name):
      return PacketDataType.from_registered_name(
          validated_graph.registered_side_packet_type_name(
              get_name(packet_tag_index_name)))

    self._side_input_type_info = {
        get_name(tag_index_name): get_side_packet_type(tag_index_name)
        for tag_index_name, _ in (side_inputs or {}).items()
    }
    return canonical_graph_config_proto

  def create_graph_options(self, options_message: message.Message,
                           values: Mapping[str, Any]) -> message.Message:
    """Sets protobuf field values."""

    if hasattr(values, 'items'):
      values = values.items()
    for pair in values:
      (field, value) = pair
      fields = field.split('.')
      m = options_message
      while len(fields) > 1:
        m = getattr(m, fields[0])
        del fields[0]
      v = getattr(m, fields[0])
      if hasattr(v, 'append'):
        del v[:]
        v.extend(value)
      elif hasattr(v, 'CopyFrom'):
        v.CopyFrom(value)
      else:
        setattr(m, fields[0], value)
    return options_message

  def _set_extension(self,
                     extension_list: containers.RepeatedCompositeFieldContainer,
                     extension_value: message.Message) -> None:
    """Sets one value in a repeated protobuf.Any extension field."""
    for extension_any in extension_list:
      if extension_any.Is(extension_value.DESCRIPTOR):
        v = type(extension_value)()
        extension_any.Unpack(v)
        v.MergeFrom(extension_value)
        extension_any.Pack(v)
        return
    extension_list.add().Pack(extension_value)

  def _make_packet(self, packet_data_type: PacketDataType,
                   data: Any) -> packet.Packet:
    if (packet_data_type == PacketDataType.IMAGE_FRAME or
        packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_creator, 'create_' + packet_data_type.value)(
          data, image_format=image_frame.ImageFormat.SRGB)
    else:
      return getattr(packet_creator, 'create_' + packet_data_type.value)(data)

  def _get_packet_content(self, packet_data_type: PacketDataType,
                          output_packet: packet.Packet) -> Any:
    """Gets packet content from a packet by type."""
    if output_packet.is_empty():
      return None
    if packet_data_type == PacketDataType.STRING:
      return packet_getter.get_str(output_packet)
    elif (packet_data_type == PacketDataType.IMAGE_FRAME or
          packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_getter, 'get_' +
                     packet_data_type.value)(output_packet).numpy_view()
    else:
      return getattr(packet_getter, 'get_' + packet_data_type.value)(
          output_packet)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
