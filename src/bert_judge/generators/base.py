from ..utils import load_hf_tokenizer


class BaseGenerator:
    def __init__(
        self,
        model_path,
        trust_remote_code=False,
        dtype="bfloat16",
        temperature=0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=2048,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.tokenizer = load_hf_tokenizer(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self._configure_tokenizer_padding()

    def _configure_tokenizer_padding(self):
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _apply_chat_template(
        self,
        prompts,
    ):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompts

        if self.model_path.split("/")[-1] == "Llama-3_3-Nemotron-Super-49B-v1_5":
            messages = [
                [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ] for prompt in prompts
            ]
        else:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        processed_prompts = []

        for _messages in messages:
            processed_prompt = self.tokenizer.apply_chat_template(
                _messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            processed_prompts.append(processed_prompt)

        return processed_prompts
