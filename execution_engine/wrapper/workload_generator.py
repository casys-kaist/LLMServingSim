from typing import List
from instruction_generator.instruction import InstructionObject
from wrapper.config_parser import ConfigObject

def generate_tensor_parallel_workload():
    pass

def generate_pipeline_parallel_workload():
    pass

def generate_hybrid_parallel_workload():
    pass

def generate_zero_inference_workload():
    # TODO: deepspeed zero-inference
    pass

def generate_workload(
        head_instruction: List[InstructionObject],
        body_instruction: List[InstructionObject],
        tail_instruction: List[InstructionObject],
        conf: ConfigObject,
        output_file: str) -> None:
    pass