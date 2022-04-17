"""Transformer pipeline."""
import os
from typing import List, Optional

import cv2

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray
from src.pipelines.chains import Chain, CombinerChain, TransformerChain


class TransformerPipeline(Transformer):
    """Pipeline Transformers"""

    def __init__(self, actuators: List[Chain]):
        """Initialize the global combiner"""
        self.actuators = actuators
        self.__assert_chain_inputs_outputs()

    def __assert_chain_inputs_outputs(self):  # pylint: disable=R0912
        """Assert that all inputs and outputs exist"""
        exist_input = False
        exist_output = False
        outputs = []
        inputs = []
        for chain in self.actuators:
            if isinstance(chain, TransformerChain):
                if chain.input_name == "input":
                    exist_input = True
                else:
                    inputs.append(chain.input_name)
                if chain.output_name == "output":
                    exist_output = True
                else:
                    outputs.append(chain.output_name)
            elif isinstance(chain, CombinerChain):
                if chain.input_name1 == "input":
                    exist_input = True
                else:
                    inputs.append(chain.input_name1)
                if chain.input_name2 == "input":
                    exist_input = True
                else:
                    inputs.append(chain.input_name2)
                if chain.output_name == "output":
                    exist_output = True
                else:
                    outputs.append(chain.output_name)
        # assert len(inputs) == len(outputs), "Inputs and outputs must be the same length"
        # assert len(inputs) == len(set(inputs)), "Inputs must be unique"
        # assert len(outputs) == len(set(outputs)), "Outputs must be unique"
        # assert sorted(inputs) == sorted(outputs), "Inputs and outputs must be the same"
        assert exist_input, "'input' chain value not found"
        assert exist_output, "'output' chain value not found"

    def __call__(
        self,
        input_img: ImageArray,
        input_cartoon: ImageArray = None,
        save_path: Optional[str] = None,
    ) -> ImageArray:
        """Applies transform to an image"""
        images = {"input": input_img}
        if input_cartoon is not None:
            images["cartoon"] = input_cartoon
        for actuator in self.actuators:
            if isinstance(actuator, TransformerChain):
                image = actuator.transformers[0](images[actuator.input_name])
                for transformer in actuator.transformers[1:]:
                    image = transformer(image)
                images[actuator.output_name] = image
            elif isinstance(actuator, CombinerChain):
                images[actuator.output_name] = actuator.combiner(
                    images[actuator.input_name1], images[actuator.input_name2]
                )

        if save_path is not None:
            self.__save_results(save_path, images["output"])
        return images["output"]

    def __save_results(self, save_path: str, image: ImageArray):
        """Saves all pipeline steps images"""
        name_from_actuators = self.__get_name_from_actuators()

        if not os.path.exists(name_from_actuators):
            os.makedirs(name_from_actuators)

        # save the image
        image_in_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(name_from_actuators, save_path), image_in_rgb)

    def __get_name_from_actuators(self) -> str:
        """Get the name from the actuators"""
        name_from_actuators = ""
        for actuator in self.actuators:
            name_from_actuators += actuator.name + "_"
        return os.path.join("data", name_from_actuators)

    def show_pipeline_steps(self):
        """Plots all pipeline steps images"""

    @staticmethod
    def show():
        """Show"""
