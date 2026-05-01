"""Llama-3-8B-Instruct victim agent (TransformerLens backend)."""

from __future__ import annotations

from typing import Optional

from .victim_base import VictimBase


LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "{system}"          # the formatted system block (or empty)
    "{turns}"           # formatted user/assistant turns
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _llama3_format(messages: list[dict], *, suffix: str = "") -> str:
    """Render a chat in Llama-3's official Instruct chat format."""
    parts: list[str] = ["<|begin_of_text|>"]
    for m in messages:
        role = m["role"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
    # Open the assistant turn that the model is being asked to complete:
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    if suffix:
        parts.append(suffix)
    return "".join(parts)


class LlamaVictim(VictimBase):
    """Llama-3-8B-Instruct via TransformerLens for full per-layer hooks."""

    def _load(self) -> None:
        import torch
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer

        # Tokenizer (HF). TransformerLens has its own but it doesn't include
        # Llama-3's chat-special tokens; we rely on string formatting instead.
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, self.dtype)
        self.model = HookedTransformer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=torch_dtype,
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.hidden_size = self.model.cfg.d_model

    # ------------------------------------------------------------------
    def _format_dialogue(self, messages: list[dict], *, suffix: str = "") -> str:
        return _llama3_format(messages, suffix=suffix)

    def _answer_prefix(self) -> str:
        # Prime the answer position so the next-token distribution is at the
        # *answer* position rather than e.g. boilerplate "Sure, ".
        return "My current answer is: "

    # ------------------------------------------------------------------
    def _run_with_hooks(self, prompt: str):
        """Run the model with TransformerLens cache to capture residuals."""
        import torch

        tokens = self.model.to_tokens(prompt, prepend_bos=False).to(self.device)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda n: n.startswith("blocks.")
                                       and n.endswith("hook_resid_post"),
                return_cache_object=True,
            )
        # Build hidden_states list: layer 0 ... layer n-1 residual streams.
        hidden_states = [
            cache["resid_post", i]                       # (1, seq, d_model)
            for i in range(self.model.cfg.n_layers)
        ]
        return logits, hidden_states

    # ------------------------------------------------------------------
    def _score_continuation(self, prompt: str, continuation: str) -> float:
        import torch

        full = prompt + continuation
        prompt_tokens = self.model.to_tokens(prompt, prepend_bos=False)
        full_tokens = self.model.to_tokens(full, prepend_bos=False)
        cont_tokens = full_tokens[:, prompt_tokens.shape[1]:]
        if cont_tokens.shape[1] == 0:
            return 0.0

        with torch.no_grad():
            logits = self.model(full_tokens.to(self.device))      # (1, seq, vocab)
        log_probs = logits.log_softmax(dim=-1)
        # Position p (0-indexed in full) predicts token at p+1.
        start = prompt_tokens.shape[1] - 1
        end = full_tokens.shape[1] - 1
        gather = log_probs[0, start:end, :].gather(
            1, cont_tokens[0].to(self.device).unsqueeze(-1)
        ).squeeze(-1)
        return float(gather.sum().item())

    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        import torch

        # TransformerLens' .generate is fine for our purposes.
        tokens = self.model.to_tokens(prompt, prepend_bos=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                tokens,
                max_new_tokens=self.gen.max_new_tokens,
                temperature=self.gen.temperature,
                top_p=self.gen.top_p,
                do_sample=self.gen.do_sample,
                stop_at_eos=True,
                eos_token_id=self.tokenizer.eos_token_id,
                verbose=False,
            )
        new = out[0, tokens.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        # Trim at the next role header if the model started one
        for marker in ("<|eot_id|>", "<|start_header_id|>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()
