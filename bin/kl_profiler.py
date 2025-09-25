"""Callback for profiling KL divergence cost during GRPO training."""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback, TrainerControl, TrainerState


_logger = logging.getLogger(__name__)


class KLProfilerCallback(TrainerCallback):
    """Computes KL statistics against a frozen reference model and logs timing."""

    def __init__(
        self,
        tokenizer,
        ref_model,
        sample_dataset=None,
        *,
        freq: int = 50,
        batch_size: int = 2,
        max_tokens: int = 256,
        seed: int = 1337,
    ) -> None:
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.sample_dataset = sample_dataset
        self.freq = max(1, freq)
        self.batch_size = max(1, batch_size)
        self.max_tokens = max(2, max_tokens)
        self._rng = random.Random(seed)
        self._last_wandb_step: Optional[int] = None
        self._warned_inputs = False

        if self.ref_model is not None:
            try:
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        step = state.global_step
        if step % self.freq != 0:
            return control

        model = kwargs.get("model")
        if model is None or self.ref_model is None:
            return control

        device = self._resolve_device(model)
        tensors = self._prepare_inputs(kwargs, device)
        if tensors is None:
            if not self._warned_inputs:
                _logger.warning("KL profiler could not obtain inputs; skipping logging.")
                self._warned_inputs = True
            return control

        input_ids, attention_mask = tensors

        start_ts = time.perf_counter()
        metrics = self._compute_metrics(model, input_ids, attention_mask)
        duration = time.perf_counter() - start_ts
        metrics["kl_profiler/duration_ms"] = duration * 1000.0
        metrics["kl_profiler/batch_size"] = float(input_ids.size(0))
        metrics["kl_profiler/seq_len_mean"] = attention_mask.sum(dim=1).float().mean().item()

        self._log_metrics(metrics, step)

        return control

    # ------------------------------------------------------------------
    def _resolve_device(self, model) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        kwargs: Dict,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        batch = kwargs.get("inputs")
        if isinstance(batch, dict) and "input_ids" in batch:
            input_ids = batch["input_ids"].detach()
            attn = batch.get("attention_mask")
            if attn is None:
                attn = (input_ids != getattr(self.tokenizer, "pad_token_id", 0)).long()
            else:
                attn = attn.detach()
            input_ids, attn = self._truncate(input_ids, attn)
            return input_ids.to(device), attn.to(device)

        if self.sample_dataset is None:
            return None

        if len(self.sample_dataset) == 0:  # pragma: no cover
            return None

        samples = [
            self.sample_dataset[int(self._rng.randrange(len(self.sample_dataset)))]
            for _ in range(self.batch_size)
        ]

        encoded = []
        for sample in samples:
            prompt = sample.get("prompt")
            if prompt is None:
                continue
            tokens = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
            )
            if not isinstance(tokens, list):
                tokens = list(tokens)
            enc_tensor = torch.tensor(tokens[: self.max_tokens], dtype=torch.long)
            encoded.append(enc_tensor)

        if not encoded:
            return None

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        padded = pad_sequence(encoded, batch_first=True, padding_value=pad_id)
        attn = (padded != pad_id).long()
        padded, attn = self._truncate(padded, attn)
        return padded.to(device), attn.to(device)

    # ------------------------------------------------------------------
    def _truncate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.size(-1) > self.max_tokens:
            input_ids = input_ids[..., : self.max_tokens].contiguous()
            attention_mask = attention_mask[..., : self.max_tokens].contiguous()
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if input_ids.size(1) < 2:
            return metrics

        def _sync():
            if input_ids.is_cuda:
                try:
                    torch.cuda.synchronize(input_ids.device)
                except Exception:
                    pass

        use_autocast = input_ids.device.type == "cuda"
        # Time policy forward
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                with torch.autocast(device_type=input_ids.device.type, enabled=use_autocast, dtype=torch.float16):
                    policy_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
        except TypeError:
            with torch.no_grad():
                with torch.autocast(device_type=input_ids.device.type, enabled=use_autocast, dtype=torch.float16):
                    policy_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
        _sync()
        metrics["kl_profiler/policy_forward_ms"] = (time.perf_counter() - t0) * 1000.0

        # Time reference forward (adapters disabled in the trainer; here ref_model is separate and frozen)
        t1 = time.perf_counter()
        try:
            with torch.no_grad():
                with torch.autocast(device_type=input_ids.device.type, enabled=use_autocast, dtype=torch.float16):
                    ref_out = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
        except TypeError:
            with torch.no_grad():
                with torch.autocast(device_type=input_ids.device.type, enabled=use_autocast, dtype=torch.float16):
                    ref_out = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
        _sync()
        metrics["kl_profiler/ref_forward_ms"] = (time.perf_counter() - t1) * 1000.0

        # Time softmax + KL compute
        t2 = time.perf_counter()
        policy_logits = policy_out.logits[:, :-1, :].to(torch.float32)
        ref_logits = ref_out.logits[:, :-1, :].to(torch.float32)
        target_mask = attention_mask[:, 1:].to(torch.float32)

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_probs = policy_log_probs.exp()

        kl = torch.sum(policy_probs * (policy_log_probs - ref_log_probs), dim=-1)
        entropy = -torch.sum(policy_probs * policy_log_probs, dim=-1)

        mask_sum = target_mask.sum(dim=-1)
        mask_sum = torch.clamp(mask_sum, min=1.0)

        kl_mean_per_seq = (kl * target_mask).sum(dim=-1) / mask_sum
        entropy_mean_per_seq = (entropy * target_mask).sum(dim=-1) / mask_sum

        metrics["kl/mean"] = kl_mean_per_seq.mean().item()
        metrics["kl/std"] = kl_mean_per_seq.std(unbiased=False).item() if kl_mean_per_seq.numel() > 1 else 0.0
        metrics["kl/entropy_mean"] = entropy_mean_per_seq.mean().item()
        metrics["kl/token_count_mean"] = mask_sum.mean().item()

        metrics["kl_profiler/softmax_kl_ms"] = (time.perf_counter() - t2) * 1000.0
        return metrics

    # ------------------------------------------------------------------
    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not metrics:
            return

        metrics["kl_profiler/global_step"] = float(step)

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                log_step = step
                if self._last_wandb_step is not None:
                    log_step = max(log_step, self._last_wandb_step + 1)
                run_step = getattr(wandb.run, "step", None)
                if run_step is not None:
                    log_step = max(log_step, int(run_step))
                wandb.log(metrics, step=log_step)
                self._last_wandb_step = log_step
        except Exception:  # pragma: no cover
            pass

        _logger.info(
            "KL profiler step %s | mean=%.5f | duration=%.1fms",
            step,
            metrics.get("kl/mean", float("nan")),
            metrics.get("kl_profiler/duration_ms", float("nan")),
        )
