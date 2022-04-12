"""Transformer pipeline."""
from typing import List

from src.base.image_array import ImageArray
from src.pipelines.chains import Chain, CombinerChain, TransformerChain


class PipelineTransformer:
    """Pipeline Trasnformers"""

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

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        images = {"input": input_img}
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
        return images["output"]

    def show_pipeline_steps(self):
        """Plots all pipeline steps images"""
