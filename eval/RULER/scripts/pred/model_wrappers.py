# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
from compass.src.load_llama import load_model, FastPrefillConfig

class HuggingFaceModel:
    def __init__(self, name_or_path: str,fastprefillconfig:FastPrefillConfig, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        if "Llama-3.1-8B-Instruct" in name_or_path:
            self.pipeline = None
            self.model,_ = load_model(fastprefillconfig,name_or_path = name_or_path)
        else:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
            except:
                self.pipeline = None
                self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]


class NanoVLLMModel:
    """
    NanoVLLM model wrapper for RULER benchmark.

    Uses nano-vllm's lightweight inference engine with CPU offload support
    for long-context inference on memory-constrained GPUs.
    """
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from nanovllm import LLM, SamplingParams
        from nanovllm.config import SparsePolicyType

        # Extract nano-vllm specific configuration
        max_model_len = generation_kwargs.pop('max_model_len', 128 * 1024)
        enable_cpu_offload = generation_kwargs.pop('enable_cpu_offload', True)
        num_gpu_blocks = generation_kwargs.pop('num_gpu_blocks', 2)
        kvcache_block_size = generation_kwargs.pop('kvcache_block_size', 1024)
        gpu_memory_utilization = generation_kwargs.pop('gpu_memory_utilization', 0.9)
        enforce_eager = generation_kwargs.pop('enforce_eager', True)

        # Build LLM kwargs
        llm_kwargs = {
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
            "kvcache_block_size": kvcache_block_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "sparse_policy": SparsePolicyType.FULL,  # Always use full attention for RULER
        }

        if enable_cpu_offload:
            llm_kwargs["enable_cpu_offload"] = True
            llm_kwargs["num_gpu_blocks"] = num_gpu_blocks

        self.llm = LLM(name_or_path, **llm_kwargs)
        self.tokenizer = self.llm.tokenizer

        # Store generation parameters
        self.stop = generation_kwargs.pop('stop', [])
        self.max_new_tokens = generation_kwargs.pop('max_new_tokens', 50)
        # NanoVLLM doesn't support greedy sampling (temperature=0.0)
        # Use a very small temperature for near-greedy behavior
        temp = generation_kwargs.pop('temperature', 0.0)
        self.temperature = max(temp, 0.01)  # Minimum 0.01 for stability

        # Pad token setup (same as HuggingFaceModel)
        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        from nanovllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

        # NanoVLLM CPU offload mode only supports single sequence
        # Process prompts one at a time
        results = []
        for prompt in prompts:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            output = outputs[0]
            text = output["text"]

            # Handle stop words
            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results
