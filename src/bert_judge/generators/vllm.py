from ..utils import (
    load_vllm_generator,
)
from .base import BaseGenerator


class vLLMGenerator(BaseGenerator):
    def __init__(
        self,
        model_path,
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=1,
        temperature=0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=2048,
    ):
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise ImportError(
                "vLLM is required for `vLLMGenerator`. Install it with `pip install vllm`."
            ) from exc

        super().__init__(
            model_path,
            trust_remote_code,
            dtype,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            max_tokens,
        )
        self.model = load_vllm_generator(
            model_path,
            trust_remote_code,
            dtype,
            tensor_parallel_size, 
        )
        self.sampling_params = SamplingParams(
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            max_tokens,
        )
                
    def generate(
        self, 
        prompts,
    ):
        prompts = self._apply_chat_template(prompts)
        prompts = self._truncate_prompts(
            prompts,
            self.model.llm_engine.model_config.max_model_len - self.max_tokens,
        )
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _truncate_prompts(
        self,
        prompts,
        max_prompt_tokens,
    ):
        max_prompt_tokens = max(1, int(max_prompt_tokens))
        truncated_prompts = []

        for prompt in prompts:
            tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_tokens,
                truncation_side="left",
            )["input_ids"]
            truncated_prompt = self.tokenizer.decode(tokens)
            truncated_prompts.append(truncated_prompt)

        return truncated_prompts
