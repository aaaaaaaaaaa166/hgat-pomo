from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BusinessEnvProfile:
    name: str
    description: str
    response_window: str | float
    delivery_window_extension: float
    resource_count: int
    order_density_ratio: float = 1.0

    def to_spec(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {
            "experiment_name": self.name,
            "delivery_window_extension": float(self.delivery_window_extension),
            "resource_count": int(self.resource_count),
            "order_density_ratio": float(self.order_density_ratio),
        }
        if self.response_window == "original":
            spec["response_window"] = "original"
        else:
            spec["response_window"] = float(self.response_window)
        return spec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BUSINESS_ENV_PROFILES: Dict[str, BusinessEnvProfile] = {
    "strict_original_env": BusinessEnvProfile(
        name="strict_original_env",
        description="Original strict dynamic response and delivery windows with one compatible resource.",
        response_window="original",
        delivery_window_extension=0.0,
        resource_count=1,
    ),
    "default_business_env": BusinessEnvProfile(
        name="default_business_env",
        description="Business-relaxed default environment validated from combined_D.",
        response_window=5.0,
        delivery_window_extension=3.0,
        resource_count=2,
    ),
}


def profile_names() -> str:
    return ",".join(sorted(BUSINESS_ENV_PROFILES))


def get_business_env_profile(name: str) -> BusinessEnvProfile:
    key = str(name).strip()
    if key not in BUSINESS_ENV_PROFILES:
        raise ValueError(f"Unknown business env profile `{name}`. Choices: {profile_names()}")
    return BUSINESS_ENV_PROFILES[key]


def apply_business_env_profile_to_args(args: argparse.Namespace, profile_name: str) -> argparse.Namespace:
    if not str(profile_name or "").strip():
        return args
    profile = get_business_env_profile(profile_name)
    args.env_profile = profile.name
    args.delivery_window_extension = float(profile.delivery_window_extension)
    args.resource_count = int(profile.resource_count)
    args.order_density_ratio = float(profile.order_density_ratio)
    if profile.response_window == "original":
        args.response_window_label = "original"
    else:
        response = float(profile.response_window)
        args.response_window = response
        args.response_window_label = str(response)
        args.response_slack_low = response
        args.response_slack_high = response
    return args
