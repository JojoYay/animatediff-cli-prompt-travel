# https://github.com/kohya-ss/sd-scripts/blob/main/networks/control_net_lllite.py

import bisect
import os
from typing import Any, List, Mapping, Optional, Type

import torch

from animatediff.utils.util import show_bytes

# input_blocksに適用するかどうか / if True, input_blocks are not applied
SKIP_INPUT_BLOCKS = False

# output_blocksに適用するかどうか / if True, output_blocks are not applied
SKIP_OUTPUT_BLOCKS = True

# conv2dに適用するかどうか / if True, conv2d are not applied
SKIP_CONV2D = False

# transformer_blocksのみに適用するかどうか。Trueの場合、ResBlockには適用されない
# if True, only transformer_blocks are applied, and ResBlocks are not applied
TRANSFORMER_ONLY = True  # if True, SKIP_CONV2D is ignored because conv2d is not used in transformer_blocks

# Trueならattn1とattn2にのみ適用し、ffなどには適用しない / if True, apply only to attn1 and attn2, not to ff etc.
ATTN1_2_ONLY = True

# Trueならattn1のQKV、attn2のQにのみ適用する、ATTN1_2_ONLY指定時のみ有効 / if True, apply only to attn1 QKV and attn2 Q, only valid when ATTN1_2_ONLY is specified
ATTN_QKV_ONLY = True

# Trueならattn1やffなどにのみ適用し、attn2などには適用しない / if True, apply only to attn1 and ff, not to attn2
# ATTN1_2_ONLYと同時にTrueにできない / cannot be True at the same time as ATTN1_2_ONLY
ATTN1_ETC_ONLY = False  # True

# transformer_blocksの最大インデックス。Noneなら全てのtransformer_blocksに適用
# max index of transformer_blocks. if None, apply to all transformer_blocks
TRANSFORMER_MAX_BLOCK_INDEX = None


class LLLiteModule(torch.nn.Module):
    def __init__(self, depth, cond_emb_dim, name, org_module, mlp_dim, dropout=None, multiplier=1.0):
        super().__init__()
        self.cond_cache ={}

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d" or org_module.__class__.__name__ == "LoRACompatibleConv"
        self.lllite_name = name
        self.cond_emb_dim = cond_emb_dim
        self.org_module = [org_module]
        self.dropout = dropout
        self.multiplier = multiplier

        if self.is_conv2d:
            in_dim = org_module.in_channels
        else:
            in_dim = org_module.in_features

        # conditioning1はconditioning imageを embedding する。timestepごとに呼ばれない
        # conditioning1 embeds conditioning image. it is not called for each timestep
        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size
        if depth == 1:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            # kernel size 8は大きすぎるので、4にする / kernel size 8 is too large, so set it to 4
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = torch.nn.Sequential(*modules)

        # downで入力の次元数を削減する。LoRAにヒントを得ていることにする
        # midでconditioning image embeddingと入力を結合する
        # upで元の次元数に戻す
        # これらはtimestepごとに呼ばれる
        # reduce the number of input dimensions with down. inspired by LoRA
        # combine conditioning image embedding and input with mid
        # restore to the original dimension with up
        # these are called for each timestep

        if self.is_conv2d:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            # midの前にconditioningをreshapeすること / reshape conditioning before mid
            self.down = torch.nn.Sequential(
                torch.nn.Linear(in_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim, in_dim),
            )

        # Zero-Convにする / set to Zero-Conv
        torch.nn.init.zeros_(self.up[0].weight)  # zero conv

        self.depth = depth  # 1~3
        self.cond_emb = None
        self.batch_cond_only = False  # Trueなら推論時のcondにのみ適用する / if True, apply only to cond at inference
        self.use_zeros_for_batch_uncond = False  # Trueならuncondのconditioningを0にする / if True, set uncond conditioning to 0

        # batch_cond_onlyとuse_zeros_for_batch_uncondはどちらも適用すると生成画像の色味がおかしくなるので実際には使えそうにない
        # Controlの種類によっては使えるかも
        # both batch_cond_only and use_zeros_for_batch_uncond make the color of the generated image strange, so it doesn't seem to be usable in practice
        # it may be available depending on the type of Control

    def _set_cond_image(self, cond_image):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        if cond_image is None:
            self.cond_emb = None
            return

        # timestepごとに呼ばれないので、あらかじめ計算しておく / it is not called for each timestep, so calculate it in advance
        # print(f"C {self.lllite_name}, cond_image.shape={cond_image.shape}")
        cx = self.conditioning1(cond_image)
        if not self.is_conv2d:
            # reshape / b,c,h,w -> b,h*w,c
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1)
        self.cond_emb = cx

    def set_cond_image(self, cond_image, cond_key):
        self.cond_image = cond_image
        self.cond_key = cond_key
        #self.cond_emb = None
        self.cond_emb = self.get_cond_emb(self.cond_key, "cuda", torch.float16)

    def set_batch_cond_only(self, cond_only, zeros):
        self.batch_cond_only = cond_only
        self.use_zeros_for_batch_uncond = zeros

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def unapply_to(self):
        self.org_module[0].forward = self.org_forward
        self.cond_cache ={}

    def get_cond_emb(self, key, device, dtype):
        #if key in self.cond_cache:
        #    return self.cond_cache[key].to(device, dtype=dtype, non_blocking=True)
        cx = self.conditioning1(self.cond_image.to(device, dtype=dtype))
        if not self.is_conv2d:
            # reshape / b,c,h,w -> b,h*w,c
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1)
        #self.cond_cache[key] = cx.to("cpu", non_blocking=True)
        return cx


    def forward(self, x, scale=1.0):
        r"""
        学習用の便利forward。元のモジュールのforwardを呼び出す
        / convenient forward for training. call the forward of the original module
        """
#        if self.multiplier == 0.0 or self.cond_emb is None:
        if (type(self.multiplier) is int and self.multiplier == 0.0) or self.cond_emb is None:
            return self.org_forward(x)

        if self.cond_emb is None:
            # print(f"cond_emb is None, {self.name}")
            '''
            cx = self.conditioning1(self.cond_image.to(x.device, dtype=x.dtype))
            if not self.is_conv2d:
                # reshape / b,c,h,w -> b,h*w,c
                n, c, h, w = cx.shape
                cx = cx.view(n, c, h * w).permute(0, 2, 1)
            #show_bytes("self.conditioning1", self.conditioning1)
            #show_bytes("cx", cx)
            '''
            self.cond_emb = self.get_cond_emb(self.cond_key, x.device, x.dtype)


        cx = self.cond_emb

        if not self.batch_cond_only and x.shape[0] // 2 == cx.shape[0]:  # inference only
            cx = cx.repeat(2, 1, 1, 1) if self.is_conv2d else cx.repeat(2, 1, 1)
            if self.use_zeros_for_batch_uncond:
                cx[0::2] = 0.0  # uncond is zero
        # print(f"C {self.lllite_name}, x.shape={x.shape}, cx.shape={cx.shape}")

        # downで入力の次元数を削減し、conditioning image embeddingと結合する
        # 加算ではなくchannel方向に結合することで、うまいこと混ぜてくれることを期待している
        # down reduces the number of input dimensions and combines it with conditioning image embedding
        # we expect that it will mix well by combining in the channel direction instead of adding
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print("cx size:", cx.size())
        print("self.down size:", self.down(x if not self.batch_cond_only else x[1::2]).size())
        
        
        cx = torch.cat([cx, self.down(x if not self.batch_cond_only else x[1::2])], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)

        if self.dropout is not None and self.training:
            cx = torch.nn.functional.dropout(cx, p=self.dropout)

        cx = self.up(cx) * self.multiplier

        #print(f"{self.multiplier=}")
        #print(f"{cx.shape=}")

        #mul = torch.tensor(self.multiplier).to(x.device, dtype=x.dtype)
        #cx = cx * mul[:,None,None]

        # residual (x) を加算して元のforwardを呼び出す / add residual (x) and call the original forward
        if self.batch_cond_only:
            zx = torch.zeros_like(x)
            zx[1::2] += cx
            cx = zx

        x = self.org_forward(x + cx)  # ここで元のモジュールを呼び出す / call the original module here
        return x




class ControlNetLLLite(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]

    def __init__(
        self,
        unet,
        cond_emb_dim: int = 16,
        mlp_dim: int = 16,
        dropout: Optional[float] = None,
        varbose: Optional[bool] = False,
        multiplier: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        # self.unets = [unet]

        def create_modules(
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
            module_class: Type[object],
        ) -> List[torch.nn.Module]:
            prefix = "lllite_unet"

            modules = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "LoRACompatibleLinear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d" or child_module.__class__.__name__ == "LoRACompatibleConv"

                        if is_linear or (is_conv2d and not SKIP_CONV2D):
                            # block indexからdepthを計算: depthはconditioningのサイズやチャネルを計算するのに使う
                            # block index to depth: depth is using to calculate conditioning size and channels
                            #print(f"{name=} {child_name=}")

                            #block_name, index1, index2 = (name + "." + child_name).split(".")[:3]
                            #index1 = int(index1)
                            block_name, num1, block_name2 ,num2 = (name + "." + child_name).split(".")[:4]

                            #if block_name == "input_blocks":
                            """
                            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
                            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."

                            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
                            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
                            """
                            if block_name == "down_blocks" and block_name2=="downsamplers":
                                index1 = 3*(int(num1)+1)
                                if SKIP_INPUT_BLOCKS:
                                    continue
                                depth = 1 if index1 <= 2 else (2 if index1 <= 5 else 3)
                            elif block_name == "down_blocks":
                                index1 = 3*int(num1)+int(num2)+1
                                if SKIP_INPUT_BLOCKS:
                                    continue
                                depth = 1 if index1 <= 2 else (2 if index1 <= 5 else 3)

                            #elif block_name == "middle_block":
                            elif block_name == "mid_block":
                                depth = 3

                            #elif block_name == "output_blocks":
                                """
                                hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
                                sd_up_res_prefix = f"output_blocks.{3*i + j}.0."

                                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                                sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
                                """
                            elif block_name == "up_blocks" and block_name2=="upsamplers":

                                index1 = 3*int(num1)+2
                                if SKIP_OUTPUT_BLOCKS:
                                    continue
                                depth = 3 if index1 <= 2 else (2 if index1 <= 5 else 1)
                                #if int(index2) >= 2:
                                if block_name2 == "upsamplers":
                                    depth -= 1
                            elif block_name == "up_blocks":
                                index1 = 3*int(num1)+int(num2)
                                if SKIP_OUTPUT_BLOCKS:
                                    continue
                                depth = 3 if index1 <= 2 else (2 if index1 <= 5 else 1)
                                #if int(index2) >= 2:
                                if block_name2 == "upsamplers":
                                    depth -= 1
                            else:
                                raise NotImplementedError()

                            lllite_name = prefix + "." + name + "." + child_name
                            lllite_name = lllite_name.replace(".", "_")

                            if TRANSFORMER_MAX_BLOCK_INDEX is not None:
                                p = lllite_name.find("transformer_blocks")
                                if p >= 0:
                                    tf_index = int(lllite_name[p:].split("_")[2])
                                    if tf_index > TRANSFORMER_MAX_BLOCK_INDEX:
                                        continue

                            #  time embは適用外とする
                            # attn2のconditioning (CLIPからの入力) はshapeが違うので適用できない
                            # time emb is not applied
                            # attn2 conditioning (input from CLIP) cannot be applied because the shape is different
                            '''
                            if "emb_layers" in lllite_name or (
                                "attn2" in lllite_name and ("to_k" in lllite_name or "to_v" in lllite_name)
                            ):
                                continue
                            '''
                            #("emb_layers.1.", "time_emb_proj."),
                            if "time_emb_proj" in lllite_name or (
                                "attn2" in lllite_name and ("to_k" in lllite_name or "to_v" in lllite_name)
                            ):
                                continue

                            if ATTN1_2_ONLY:
                                if not ("attn1" in lllite_name or "attn2" in lllite_name):
                                    continue
                                if ATTN_QKV_ONLY:
                                    if "to_out" in lllite_name:
                                        continue

                            if ATTN1_ETC_ONLY:
                                if "proj_out" in lllite_name:
                                    pass
                                elif "attn1" in lllite_name and (
                                    "to_k" in lllite_name or "to_v" in lllite_name or "to_out" in lllite_name
                                ):
                                    pass
                                elif "ff_net_2" in lllite_name:
                                    pass
                                else:
                                    continue

                            module = module_class(
                                depth,
                                cond_emb_dim,
                                lllite_name,
                                child_module,
                                mlp_dim,
                                dropout=dropout,
                                multiplier=multiplier,
                            )
                            modules.append(module)
            return modules

        target_modules = ControlNetLLLite.UNET_TARGET_REPLACE_MODULE
        if not TRANSFORMER_ONLY:
            target_modules = target_modules + ControlNetLLLite.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        # create module instances
        self.unet_modules: List[LLLiteModule] = create_modules(unet, target_modules, LLLiteModule)
        print(f"create ControlNet LLLite for U-Net: {len(self.unet_modules)} modules.")

    def forward(self, x):
        return x  # dummy

    def set_cond_image(self, cond_image, cond_key):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        for module in self.unet_modules:
            module.set_cond_image(cond_image,cond_key)

    def set_batch_cond_only(self, cond_only, zeros):
        for module in self.unet_modules:
            module.set_batch_cond_only(cond_only, zeros)

    def set_multiplier(self, multiplier):
        if isinstance(multiplier, list):
            multiplier = torch.tensor(multiplier).to("cuda", dtype=torch.float16, non_blocking=True)
            multiplier = multiplier[:,None,None]

        for module in self.unet_modules:
            module.multiplier = multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self):
        print("applying LLLite for U-Net...")
        for module in self.unet_modules:
            module.apply_to()
            self.add_module(module.lllite_name, module)

    def unapply_to(self):
        for module in self.unet_modules:
            module.unapply_to()

    # マージできるかどうかを返す
    def is_mergeable(self):
        return False

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        raise NotImplementedError()

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self):
        self.requires_grad_(True)
        return self.parameters()

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_epoch_start(self):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        from animatediff.utils.lora_diffusers import UNET_CONVERSION_MAP

        # convert SDXL Stability AI's state dict to Diffusers' based state dict
        map_keys = list(UNET_CONVERSION_MAP.keys())  # prefix of U-Net modules
        map_keys.sort()
        for key in list(state_dict.keys()):
            if key.startswith("lllite_unet" + "_"):
                search_key = key.replace("lllite_unet" + "_", "")
                position = bisect.bisect_right(map_keys, search_key)
                map_key = map_keys[position - 1]
                if search_key.startswith(map_key):
                    new_key = key.replace(map_key, UNET_CONVERSION_MAP[map_key])
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        # in case of V2, some weights have different shape, so we need to convert them
        # because V2 LoRA is based on U-Net created by use_linear_projection=False
        my_state_dict = self.state_dict()
        for key in state_dict.keys():
            if state_dict[key].size() != my_state_dict[key].size():
                # print(f"convert {key} from {state_dict[key].size()} to {my_state_dict[key].size()}")
                state_dict[key] = state_dict[key].view(my_state_dict[key].size())

        return super().load_state_dict(state_dict, strict)


def load_controlnet_lllite(model_file, pipe, torch_dtype=torch.float16):
    print(f"loading ControlNet-LLLite: {model_file}")

    from safetensors.torch import load_file

    state_dict = load_file(model_file)
    mlp_dim = None
    cond_emb_dim = None
    for key, value in state_dict.items():
        if mlp_dim is None and "down.0.weight" in key:
            mlp_dim = value.shape[0]
        elif cond_emb_dim is None and "conditioning1.0" in key:
            cond_emb_dim = value.shape[0] * 2
        if mlp_dim is not None and cond_emb_dim is not None:
            break
    assert mlp_dim is not None and cond_emb_dim is not None, f"invalid control net: {model_file}"

    control_net = ControlNetLLLite(pipe.unet, cond_emb_dim, mlp_dim, multiplier=1.0)
    control_net.apply_to()
    info = control_net.load_state_dict(state_dict, False)
    print(info)
    #control_net.to(dtype).to(device)
    control_net.to(torch_dtype)
    control_net.set_batch_cond_only(False, False)
    return control_net
