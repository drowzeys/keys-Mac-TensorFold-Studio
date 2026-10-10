"""ComfyUI node: FastH3's transformer blocks on TensorFold's CUDA engine (int8 on the tensor cores).

ComfyUI keeps the text encoder, the sampler, the patch projections, the final layer and the decoders. This node
replaces the 50 blocks of each denoising pass with one call into libtf_h3.so (TensorFold's H3 family, built from
zig/src/families/h3/cuda.zig), which reads the same checkpoint file and quantizes it on the GPU when first used.

Environment: TF_H3_LIB (the library; default beside this file), TF_H3_CHECK=1 (first pass of a run: also run
ComfyUI's own blocks and log how close the engine is), TF_H3_PROFILE=1 (log where a pass's time went).
"""

import ctypes
import logging
import math
import os
import time

import torch

import comfy.patcher_extension
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model

TILE = 64
CUBE = (4, 4, 4)


class Inputs(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("tables", ctypes.c_uint64), ("line", ctypes.c_uint64), ("cos", ctypes.c_uint64),
                ("sin", ctypes.c_uint64), ("slot", ctypes.c_uint64), ("sizes", ctypes.c_uint64), ("rows", ctypes.c_uint32),
                ("lines", ctypes.c_uint32), ("rot", ctypes.c_uint32), ("tiles", ctypes.c_uint32),
                ("prefix_tiles", ctypes.c_uint32), ("keep", ctypes.c_uint32), ("blocks", ctypes.c_uint32),
                ("attention_out", ctypes.c_uint64)]


_library = None
_engines = {}   # checkpoint path -> engine handle; the int8 weights stay on the GPU while ComfyUI runs


def library():
    global _library
    if _library is None:
        path = os.environ.get("TF_H3_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtf_h3.so")
        lib = ctypes.CDLL(path)
        lib.tf_h3_open.restype = ctypes.c_void_p
        lib.tf_h3_open.argtypes = [ctypes.c_char_p]
        lib.tf_h3_close.argtypes = [ctypes.c_void_p]
        lib.tf_h3_shape.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32 * 7)]
        lib.tf_h3_forward.restype = ctypes.c_double
        lib.tf_h3_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(Inputs)]
        lib.tf_h3_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.tf_h3_spent.restype = ctypes.c_int
        lib.tf_h3_spent.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double * 16), ctypes.POINTER(ctypes.c_char_p * 16)]
        _library = lib
    return _library


def engine(path):
    handle = _engines.get(path)
    if handle is None:
        torch.cuda.synchronize()
        handle = library().tf_h3_open(path.encode())
        if not handle:
            raise RuntimeError(f"TensorFold H3: could not load {path}")
        shape = (ctypes.c_uint32 * 7)()
        library().tf_h3_shape(handle, ctypes.byref(shape))
        logging.info(f"TensorFold H3: {shape[4]} blocks of {shape[0]}/{shape[1]}/{shape[3]} in int8, "
                     f"{'read from the saved copy' if shape[6] else 'converted from the checkpoint (saved for the next start)'} in {shape[5] / 1000:.1f} s")
        _engines[path] = handle
    return handle


def release():
    for handle in _engines.values():
        library().tf_h3_close(handle)
    _engines.clear()


def tile_plan(layout, device):
    """FastVideo's tile order: every segment ahead of the video in its own 64-row tiles, the video in 4x4x4 cubes.

    Returns each row's padded slot, each tile's real rows, and the number of prefix tiles.
    """
    _text, latent_t, latent_h, latent_w, _audio = layout.signature
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    tiles, prefix = [], 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind != "video":
            m = (n + TILE - 1) // TILE
            seg = torch.full((m * TILE,), -1, dtype=torch.int64)
            seg[:n] = torch.arange(a, b)
            tiles.append(seg.view(m, TILE))
            prefix += m
            continue
        if grid[0] * grid[1] * grid[2] != n:
            raise RuntimeError(f"TensorFold H3: video segment of {n} rows does not match the latent grid {grid}")
        pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, CUBE))
        padded = torch.full((pt, ph, pw), -1, dtype=torch.int64)
        padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b).view(*grid)
        cubes = (padded.view(pt // CUBE[0], CUBE[0], ph // CUBE[1], CUBE[1], pw // CUBE[2], CUBE[2])
                 .permute(0, 2, 4, 1, 3, 5).reshape(-1, TILE))
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiles.append(torch.gather(cubes, 1, order))
    tiles = torch.cat(tiles)
    src = tiles.reshape(-1)
    live = src >= 0
    slot = torch.empty(layout.seq_len, dtype=torch.int64)
    slot[src[live]] = torch.nonzero(live).flatten()
    return (slot.to(torch.int32).to(device), (tiles >= 0).sum(1).to(torch.int32).to(device), prefix)


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


class Blocks:
    """One patched model's state: the engine, and what stays the same through a sampling run."""

    def __init__(self, diffusion_model, path, sparsity):
        self.model = diffusion_model
        self.path = path
        self.sparsity = sparsity
        self.plans = {}
        self.checked = False
        self.passes = 0

    def plan(self, layout, device):
        key = (tuple(layout.signature), tuple(layout.segments))
        if key not in self.plans:
            self.plans = {key: tile_plan(layout, device)}
        return self.plans[key]

    def run(self, args, blocks=0, dense=False):
        h, layout = args["img"], args["layout"]
        device = h.device
        rows = h.shape[0]
        if not h.is_contiguous() or h.dtype != torch.bfloat16:
            raise RuntimeError(f"TensorFold H3 wants a contiguous bf16 stream, got {h.dtype}")
        tables = torch.stack([torch.stack(block.adaln_proj(args["t_emb"]), dim=1) for block in self.model.blocks])
        tables = tables.to(torch.float32).contiguous()                      # (blocks, lines, 6, hidden)
        line = torch.empty(rows, dtype=torch.int32, device=device)
        for a, b, row in args["mod_segments"]:
            line[a:b] = row if isinstance(row, int) else row.to(torch.int32)
        rope = args["rope_freqs"]                                            # (1, rows, 1, rot, 2, 2): c, -s, s, c
        cos = rope[0, :, 0, :, 0, 0].to(torch.float32).contiguous()
        sin = rope[0, :, 0, :, 1, 0].to(torch.float32).contiguous()
        slot, sizes, prefix = self.plan(layout, device)
        video_tiles = sizes.shape[0] - prefix
        keep = 0 if dense or self.sparsity <= 0 else max(1, min(math.ceil((1.0 - self.sparsity) * video_tiles), video_tiles))
        inputs = Inputs(x=h.data_ptr(), tables=tables.data_ptr(), line=line.data_ptr(), cos=cos.data_ptr(), sin=sin.data_ptr(),
                        slot=slot.data_ptr(), sizes=sizes.data_ptr(), rows=rows, lines=tables.shape[1], rot=cos.shape[1],
                        tiles=sizes.shape[0], prefix_tiles=prefix, keep=keep, blocks=blocks, attention_out=0)
        handle = engine(self.path)
        # the engine's kernels go on the default stream: whatever torch has queued on its own must have landed
        torch.cuda.synchronize()
        took = library().tf_h3_forward(handle, ctypes.byref(inputs))
        if took < 0:
            raise RuntimeError("TensorFold H3: the forward failed")
        return took, keep, sizes.shape[0]

    def check(self, args):
        """The engine against ComfyUI's own blocks on this pass's stream: after one block and after all of them."""
        start = args["img"].clone()
        options = args["transformer_options"]

        def reference(count):
            h = start.clone()
            for block in self.model.blocks[:count]:
                h = block(h, args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
            return h

        for count in (1, len(self.model.blocks)):
            want = reference(count)
            for label, dense in (("dense", True), ("sparse", False)):
                if not dense and self.sparsity <= 0:
                    continue
                got = start.clone()
                self.run({**args, "img": got}, blocks=count, dense=dense)
                error = float((got.float() - want.float()).norm() / want.float().norm())
                logging.info(f"TensorFold H3 check: {count} block(s), engine {label} against ComfyUI dense: "
                             f"cosine {cosine(got, want):.5f}, relative error {error:.4f}")
            del want

    def first(self, args, extra):
        if os.environ.get("TF_H3_CHECK") == "1" and not self.checked:
            self.checked = True
            self.check(args)
        profile = os.environ.get("TF_H3_PROFILE") == "1"
        handle = engine(self.path)
        if profile:
            library().tf_h3_profile(handle, 1)
        mark = time.perf_counter()
        took, keep, tiles = self.run(args)
        self.passes += 1
        if self.passes == 1:
            mode = f"{keep} video tiles kept a query tile" if keep else "dense attention"
            logging.info(f"TensorFold H3: {args['img'].shape[0]} rows in {tiles} tiles, {mode}, first pass {took:.2f} s")
        if profile:
            spent, names = (ctypes.c_double * 16)(), (ctypes.c_char_p * 16)()
            n = library().tf_h3_spent(handle, ctypes.byref(spent), ctypes.byref(names))
            parts = ", ".join(f"{names[i].decode()} {spent[i]:.2f}" for i in range(n))
            logging.info(f"TensorFold H3 pass {self.passes}: {time.perf_counter() - mark:.2f} s ({parts})")
            library().tf_h3_profile(handle, 0)
        return {"img": args["img"]}


def passthrough(args, extra):
    return {"img": args["img"]}


class TensorFoldH3Blocks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The FastH3 bf16 checkpoint the model was loaded from."}),
            "sparsity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 0.99, "step": 0.05,
                                   "tooltip": "FastH3's sparse attention: the share of video tiles a tile leaves out. 0 is dense."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"

    def patch(self, model, unet_name, sparsity):
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("TensorFold H3 needs a MiniMax H3 / FastH3 model")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        state = Blocks(diffusion_model, path, sparsity)
        m = model.clone()
        for i in range(len(diffusion_model.blocks)):
            m.set_model_patch_replace(state.first if i == 0 else passthrough, "dit", "double_block", i)

        def no_prefetch(executor, x, timestep, context, transformer_options={}, **kwargs):
            # ComfyUI would stream its own copy of every block's weights onto the GPU ahead of the block
            transformer_options["prefetch_dynamic_vbars"] = False
            return executor(x, timestep, context, transformer_options, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "tensorfold_h3", no_prefetch)

        def reset(model_patcher):
            state.checked = False
            state.passes = 0

        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "tensorfold_h3", reset)
        return (m,)


NODE_CLASS_MAPPINGS = {"TensorFoldH3Blocks": TensorFoldH3Blocks}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorFoldH3Blocks": "TensorFold H3 Blocks (CUDA int8)"}
