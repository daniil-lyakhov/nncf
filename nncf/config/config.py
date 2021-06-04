"""
 Copyright (c) 2019 Intel Corporation
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

import itertools

from copy import deepcopy
from pathlib import Path
from typing import List, Type

import jsonschema
import jstyleson as json

from nncf.common.utils.logger import logger
from nncf.common.utils.os import safe_open
from nncf.config.schema import ROOT_NNCF_CONFIG_SCHEMA
from nncf.config.schema import validate_single_compression_algo_schema
from nncf.config.structure import NNCFExtraConfigStruct

DEFAULT_TARGET_DEVICE = 'ANY'


class NNCFConfig(dict):
    """A regular dictionary object extended with some utility functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__nncf_extra_structs = {}  # type: dict[str, NNCFExtraConfigStruct]

    @classmethod
    def from_dict(cls, nncf_dict):
        """
        Load NNCF config from dict;
        The dict must contain only json supported primitives.
        """

        NNCFConfig.validate(nncf_dict)
        nncf_config = cls(deepcopy(nncf_dict))
        nncf_config._configure_algo_params()
        return nncf_config

    @classmethod
    def from_json(cls, path) -> 'NNCFConfig':
        file_path = Path(path).resolve()
        with safe_open(file_path) as f:
            loaded_json = json.load(f)
        return cls.from_dict(loaded_json)

    def register_extra_structs(self, struct_list: List[NNCFExtraConfigStruct]):
        for struct in struct_list:
            struct_id = struct.get_id()
            if struct_id in self.__nncf_extra_structs:
                raise RuntimeError("{} is already registered as extra struct in NNCFConfig!")
            self.__nncf_extra_structs[struct_id] = struct

    def get_extra_struct(self, struct_cls: Type[NNCFExtraConfigStruct]) -> NNCFExtraConfigStruct:
        return self.__nncf_extra_structs[struct_cls.get_id()]

    def get_all_extra_structs_for_copy(self) -> List[NNCFExtraConfigStruct]:
        return list(self.__nncf_extra_structs.values())

    @staticmethod
    def validate(loaded_json):
        try:
            jsonschema.validate(loaded_json, schema=ROOT_NNCF_CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.error("Invalid NNCF config supplied!")

            # The default exception's __str__ result will contain the entire schema,
            # which is too large to be readable.
            import nncf.config.schema as config_schema
            msg = e.message + ". See documentation or {} for an NNCF configuration file JSON schema definition".format(
                config_schema.__file__)
            raise jsonschema.ValidationError(msg)

        compression_section = loaded_json.get("compression")
        if compression_section is None:
            # No compression specified
            return

        try:
            if isinstance(compression_section, dict):
                validate_single_compression_algo_schema(compression_section)
            else:
                # Passed a list of dicts
                for compression_algo_dict in compression_section:
                    validate_single_compression_algo_schema(compression_algo_dict)
        except jsonschema.ValidationError:
            # No need to trim the exception output here since only the compression algo
            # specific sub-schema will be shown, which is much shorter than the global schema
            logger.error("Invalid NNCF config supplied!")
            raise

    def _configure_algo_params(self):
        algorithm_configs = self.get('compression')
        if not algorithm_configs:
            return

        if isinstance(algorithm_configs, dict):
            algorithm_configs = [algorithm_configs]

        hw_config_type = None
        target_device = self.get('target_device', 'ANY')
        compression_lr_multiplier = self.get('compression_lr_multiplier', None)
        if target_device != 'TRIAL':
            from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
            from nncf.common.hardware.config import HWConfigType
            hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])


        modified_algo_configs = []
        for algo_conf in algorithm_configs:
            modified_algo_conf = NNCFConfig(algo_conf)
            modified_algo_conf.register_extra_structs(self.get_all_extra_structs_for_copy())
            modified_algo_conf.update({
                'target_device': target_device,
                'compression_lr_multiplier': compression_lr_multiplier,
                'hw_config_type': hw_config_type
            })
            modified_algo_configs.append(modified_algo_conf)

        # TODO: We can keep list with one element and use
        # len(config.get(compression)) instead of isinstance
        if len(algorithm_configs) == 1:
            modified_algo_configs = modified_algo_configs[0]

        self['compression'] = modified_algo_configs


def product_dict(d):
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
