from dataclasses import dataclass, asdict


@dataclass
class Config:
    prompt: str = "a photo of sandwich with ham, cheese, and cucumber"
    negative_prompt: str = ""
    manual_seed: int or None = None
    seeds_no: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 40
    is_step_by_step: bool = False
    is_nsfw: bool = False

    def as_dict(self) -> dict:
        return asdict(self)
