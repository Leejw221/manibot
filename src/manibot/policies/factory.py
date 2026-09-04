import importlib
import logging
from enum import Enum
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    POLICY = "policy"

    @classmethod
    def from_str(cls, value: str) -> 'ComponentType':
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown component type: {value}")


class Registry:
    """Registry for dynamically loading components"""

    def __init__(self):
        # Initialize registry containers for different component types
        self._components = {ct: {} for ct in ComponentType}

    def register_component(self, component_type: ComponentType, name: str):
        """Generic decorator to register a component class"""
        def _register(cls):
            self._components[component_type][name] = cls
            return cls
        return _register

    def get_component(self, component_type: ComponentType, name: str):
        """Get component class by type and name"""
        components = self._components[component_type]
        if name not in components:
            raise ValueError(f"Unknown {component_type.value}: {name}")
        return components[name]

    def list_components(self, component_type: ComponentType) -> list[str]:
        """List all registered components of a given type"""
        return list(self._components[component_type].keys())

    # Convenience methods for common component types
    def register_policy(self, name: str):
        """Decorator to register a policy class"""
        return self.register_component(ComponentType.POLICY, name)

    def get_policy(self, name: str):
        """Get policy class by name"""
        return self.get_component(ComponentType.POLICY, name)

    def list_policies(self) -> list[str]:
        """List all registered policies"""
        return self.list_components(ComponentType.POLICY)

    def register_from_module(self, component_type: ComponentType, module_path: str, suffix: str = ''):
        """Register components from a module path

        Args:
            component_type: Type of component from ComponentType enum
            module_path: Path to the module (e.g., 'manibot.policies')
            suffix: Suffix to strip from class names (e.g., 'policy')
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Could not import module {module_path}: {e}")
            return self

        register_method = getattr(self, f"register_{component_type.value}", None)
        if register_method is None:
            def register_method(name): return self.register_component(component_type, name)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Skip base/abstract classes (e.g., BasePolicy)
            if attr_name.startswith('Base'):
                continue

            # Check if it's a class with the right suffix
            if isinstance(attr, type) and attr_name.lower().endswith(suffix.lower()):
                name = attr_name[:-(len(suffix))] if suffix else attr_name
                name = name.lower()
                register_method(name)(attr)
                logger.debug(f"Registered {component_type.value}: {name}")

        return self


# Create a global registry instance
registry = Registry()

# Register built-in components
registry.register_from_module(ComponentType.POLICY, 'manibot.policies', 'policy')


def create_policy(cfg: DictConfig) -> torch.nn.Module:
    policy_name = cfg.policy.name
    policy_cls = registry.get_policy(policy_name)
    policy = policy_cls(cfg).to(cfg.device)
    return policy


def get_policy_class(name: str):
    return registry.get_policy(name)


# ---------------------------------------------------------------------------
# LeRobot policies
#
# The diffusion policy comes from LeRobot rather than from a copy of our own:
# LeRobot is maintained (we already had to follow one rename, lerobot.types ->
# lerobot.lerobot_types), its DiffusionPolicy is the reference implementation
# other people's numbers are comparable to, and it costs us no coupling — it
# imports only lerobot.utils.constants, whose OBS_STATE / ACTION are exactly the
# observation names this repository already standardised on.
#
# One shape difference: LeRobot normalizes outside the policy (a processor
# pipeline) while our older policies normalize inside. make_policy therefore
# always returns a (policy, preprocessor, postprocessor) triple, with identity
# processors for the policies that normalize internally, so the training loop
# has a single contract to speak.
# ---------------------------------------------------------------------------


def _identity(x):
    return x


def _lerobot_diffusion(cfg, dataset_meta, stats):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

    task, pol = cfg.task, cfg.policy
    sched, unet = pol.noise_scheduler, pol.unet

    def visual_shape(key):
        # config.json stores images HWC; LeRobot policy features are CHW.
        h, w, c = dataset_meta.features[key]["shape"]
        return (c, h, w)

    lr_cfg = DiffusionConfig(
        n_obs_steps=pol.obs_horizon,
        horizon=pol.pred_horizon,
        n_action_steps=pol.action_horizon,
        input_features={
            task.state_key: PolicyFeature(type=FeatureType.STATE, shape=(task.state_dim,)),
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=visual_shape(k))
               for k in task.image_keys},
        },
        output_features={
            task.action_key: PolicyFeature(type=FeatureType.ACTION, shape=(task.action_dim,)),
        },
        resize_shape=tuple(cfg.resize_shape) if cfg.get("resize_shape") else None,
        crop_shape=tuple(cfg.crop_shape) if cfg.get("crop_shape") else None,
        crop_is_random=pol.get("crop_is_random", True),
        spatial_softmax_num_keypoints=pol.get("spatial_softmax_num_keypoints", 32),
        use_separate_rgb_encoder_per_camera=pol.get("use_separate_rgb_encoder_per_camera", True),
        # LeRobot 기본값은 ImageNet 사전학습 ResNet18 + BatchNorm 인데, 원본 Diffusion
        # Policy 구현(과 우리가 여태 쓰던 인코더)은 scratch ResNet18 + GroupNorm 이다.
        # 사전학습 가중치를 둔 채로 GroupNorm 으로 바꾸면 그 가중치가 망가지므로
        # LeRobot 이 아예 거부한다 — 둘은 같이 못 쓴다.
        use_group_norm=pol.get("use_group_norm", True),
        pretrained_backbone_weights=pol.get("pretrained_backbone_weights", None),
        down_dims=tuple(unet.down_dims),
        kernel_size=unet.kernel_size,
        n_groups=unet.n_groups,
        diffusion_step_embed_dim=unet.diffusion_step_embed_dim,
        use_film_scale_modulation=unet.use_film_scale_modulation,
        noise_scheduler_type=sched.type,
        num_train_timesteps=sched.num_train_timesteps,
        beta_schedule=sched.beta_schedule,
        beta_start=sched.get("beta_start", 0.0001),
        beta_end=sched.get("beta_end", 0.02),
        prediction_type=sched.prediction_type,
        clip_sample=sched.get("clip_sample", True),
        clip_sample_range=sched.get("clip_sample_range", 1.0),
        num_inference_steps=sched.num_inference_steps,
        optimizer_lr=cfg.optimizer_lr,
        optimizer_betas=tuple(cfg.optimizer_betas),
        optimizer_eps=cfg.optimizer_eps,
        optimizer_weight_decay=cfg.optimizer_weight_decay,
        scheduler_name=cfg.scheduler_name,
        scheduler_warmup_steps=cfg.scheduler_warmup_steps,
        device=str(cfg.device),
        push_to_hub=False,
    )
    # The processors normalize; they need tensors, while config.json holds lists.
    tensor_stats = {
        k: {m: torch.as_tensor(v, dtype=torch.float32) for m, v in s.items()}
        for k, s in stats.items()
    }
    policy = DiffusionPolicy(lr_cfg)
    pre, post = make_diffusion_pre_post_processors(lr_cfg, tensor_stats)
    return policy, pre, post


def make_policy(cfg, dataset_meta, stats):
    """Build (policy, preprocessor, postprocessor) for cfg.policy.name."""
    if cfg.policy.name == "diffusion":
        return _lerobot_diffusion(cfg, dataset_meta, stats)
    # act · cfm still carry their normalization inside the module.
    policy = get_policy_class(cfg.policy.name)(cfg, stats)
    return policy, _identity, _identity
