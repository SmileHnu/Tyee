#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : base.py
@Time    : 2024/11/04 15:32:25
@Desc    : 
"""
import sys
import torch
from torch import nn
from typing import Callable


class Hook:
    def __init__(
        self,
        module_path: str, 
        transform: Callable, 
        unique_identifier: str = None
    ) -> None:
        """
        Hook for capturing intermediate hidden states in a model layer.
        :param str module_path: which layer to hook.
        :param Callable transform: unction to transform the hidden state.
        :param str unique_identifier: Unique name for identifying this hook, defaults to None
        """
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)

    def register(self, module: nn.Module, hiddens: list[str, torch.Tensor]):
        """ register hook """
        self.handler = module.register_forward_hook(lambda mod, inp, out: hiddens.append((self.unique_identifier, self.transform(inp, out))))

    def unregister(self) -> None:
        """ unregister this hook if it is active """
        if self.handler:
            self.handler.remove()
            self.handler = None
        pass


class BaseExpert(nn.Module):
    def __init__(
        self,
        hooks: list[tuple] = None,
        hook_postprocess: Callable[
            [list[tuple[str, torch.Tensor]]], list[tuple[str, torch.Tensor]]
        ] = None,
        **kwargs,
    ) -> None:
        """
        Base model expert with hooks to capture hidden states.
        :param list[tuple] hooks: List of hooks to initialize.
        :param Callable[ [list[tuple[str, torch.Tensor]]], list[tuple[str, torch.Tensor]] ] hook_postprocess: Postprocessing function for hook outputs.
        """
        super().__init__()
        self.hooks: list[Hook] = [Hook(*hook) for hook in hooks] if hooks else []
        self.hook_postprocess = hook_postprocess
        # the hidden states of hooks
        self._hook_hiddens: list[tuple[str, torch.Tensor]] = []

    def remove_all_hooks(self) -> None:
        """ remove all hooks """
        for hook in self.hooks:
            hook.unregister()
        self.hooks.clear()

    def remove_hook(self, unique_identifier: str):
        """
        remove hook by unique_identifier
        :param str unique_identifier: the hook's unique identifier
        """
        updated_hooks = []
        for hook in self.hooks:
            if hook.unique_identifier == unique_identifier:
                hook.unregister()
            else:
                updated_hooks.append(hook)
        self.hooks = updated_hooks

    def add_hook(self, *args, **kwargs):
        """ add hook """
        hook = Hook(*args, **kwargs)
        
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):
            print(f"{hook.module_path} is not a valid nn.Module. Skip.", file=sys.stderr)
            return
    
        # 该 hook 已被注册过了，则删除并重新注册
        if callable(hook.handler):
            print(f"Existing hook handler for {hook.unique_identifier} is found. Remove the existing one.", file=sys.stderr)
            hook.unregister()
        
        hook.register(module=module, hiddens=self._hook_hiddens)

        self.hooks.append(hook)


    def __call__(self, wavs: list[torch.Tensor], *args, **kwargs):
        self._hook_hiddens.clear()

        # forward pass
        result = super().__call__(wavs, *args, **kwargs) or {}

        assert isinstance(result, dict)

        if len(self._hook_hiddens) > 0:
            if (
                result.get("_hidden_states_info") is not None
                or result.get("hidden_states") is not None
                or result.get("last_hidden_state") is not None
            ):
                print(
                    "If there are registered hooks, '_hidden_states_info', 'hidden_states', and "
                    "'last_hidden_state' are reserved and should not be included in child class's return dict.",
                    file=sys.stderr,
                )
                raise ValueError

            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            result["_hidden_states_info"], result["hidden_states"] = zip(*hook_hiddens)
            result["last_hidden_state"] = result["hidden_states"][-1]

            for layer_id, hidden_state in enumerate(result["hidden_states"]):
                result[f"hidden_state_{layer_id}"] = hidden_state

            return result["last_hidden_state"]
        else:
            if "x" in result:
                return result["x"]
            else:
                raise ValueError("No results from hooks.")

    def get_downsample_rates(self) -> int:
        """ get downsample rates """
        raise NotImplementedError
