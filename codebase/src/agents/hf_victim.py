"""Shared HuggingFace + native-PyTorch-hook backend for non-TL victims.

TransformerLens does not officially support Pythia-6.9B or Mistral-7B-v0.1
checkpoints (loading is finicky and breaks for several minor architectural
quirks). Instead, we use plain HF Transformers and register forward hooks on
each transformer block to capture per-layer residual stream outputs.

``HFVictimBase`` factors out the load + activation hooks. Subclasses define
the chat format only.
"""

from __future__ import annotations

from .victim_base import VictimBase


class HFVictimBase(VictimBase):
    """Common implementation for any HF causal-LM with `model.<layers>` blocks."""

    # Subclasses override this to point to the right attribute path on the model
    # (e.g. "model.layers" for Mistral, "gpt_neox.layers" for Pythia).
    LAYERS_ATTR: str = "model.layers"

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
        )
        self.model.eval()

        # Resolve the layer list (works for Pythia and Mistral via overrides).
        layers = self._get_layers_list()
        self.n_layers = len(layers)
        # hidden size: prefer model.config
        cfg = self.model.config
        self.hidden_size = getattr(
            cfg, "hidden_size",
            getattr(cfg, "d_model", getattr(cfg, "n_embd", 0)),
        )

    # ------------------------------------------------------------------
    def _get_layers_list(self):
        """Resolve the per-layer module list using ``LAYERS_ATTR`` (dotted)."""
        obj = self.model
        for part in self.LAYERS_ATTR.split("."):
            obj = getattr(obj, part)
        return obj

    # ------------------------------------------------------------------
    def _run_with_hooks(self, prompt: str):
        import torch

        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        layers = self._get_layers_list()
        captured: list = [None] * len(layers)

        def make_hook(idx):
            def hook(module, inputs, output):
                # Block outputs may be a tensor or a tuple (hidden, ...).
                hs = output[0] if isinstance(output, tuple) else output
                captured[idx] = hs
            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
        try:
            with torch.no_grad():
                out = self.model(**toks, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()

        return out.logits, captured

    # ------------------------------------------------------------------
    def _score_continuation(self, prompt: str, continuation: str) -> float:
        import torch

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = self.tokenizer(prompt + continuation, return_tensors="pt").input_ids
        cont_ids = full_ids[:, prompt_ids.shape[1]:]
        if cont_ids.shape[1] == 0:
            return 0.0

        with torch.no_grad():
            logits = self.model(full_ids.to(self.device)).logits
        log_probs = logits.log_softmax(dim=-1)
        start = prompt_ids.shape[1] - 1
        end = full_ids.shape[1] - 1
        gather = log_probs[0, start:end, :].gather(
            1, cont_ids[0].to(self.device).unsqueeze(-1)
        ).squeeze(-1)
        return float(gather.sum().item())

    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        import torch

        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **toks,
                max_new_tokens=self.gen.max_new_tokens,
                temperature=self.gen.temperature,
                top_p=self.gen.top_p,
                do_sample=self.gen.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0, toks.input_ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return self._trim_after_generation(text).strip()

    # Subclasses can override (e.g. trim "Critic:" continuations).
    def _trim_after_generation(self, text: str) -> str:
        return text
