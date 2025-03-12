import ipdb
from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type
def module_active_adapters(module):
    if hasattr(module, 'active_adapters'):
        result = [i for i in module.active_adapters if i in module.scaling.keys()]
    else:
        result = []
    return result

class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], enable_adapters: List) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.active_adapter_scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in module_active_adapters(lora_module)
            }
            for lora_module in self.lora_modules
        ]
        self.enable_adapters = enable_adapters

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            for active_adapter in module_active_adapters(lora_module):
                if active_adapter not in self.enable_adapters:
                    lora_module.set_scale(active_adapter,0)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            for active_adapter in module_active_adapters(lora_module):
                lora_module.set_scale(active_adapter,self.active_adapter_scales[i][active_adapter])
