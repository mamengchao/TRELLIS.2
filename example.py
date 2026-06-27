"""
TRELLIS.2 图片转3D 命令行工具
===============================
使用示例:
    python example.py --image assets/example_image/demo.webp --seed 42
    python example.py --pipeline-type 1536_cascade --tex-steps 20 --output-dir ./my_output
"""
import os
import argparse
import json
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


# --- Default configuration path ---
DEFAULT_CONFIG_PATH = (
    "E:/huggingface-cache/hub/models--microsoft--TRELLIS.2-4B"
    "/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/pipeline.json"
)


def load_config(config_path: str) -> dict:
    """Load pipeline.json and extract default sampler parameters."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    args = cfg.get('args', {})
    defaults = {
        'pipeline_type': args.get('default_pipeline_type', '1024_cascade'),
        'sparse_structure_sampler_params': args.get('sparse_structure_sampler', {}).get('params', {}),
        'shape_slat_sampler_params': args.get('shape_slat_sampler', {}).get('params', {}),
        'tex_slat_sampler_params': args.get('tex_slat_sampler', {}).get('params', {}),
    }
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 图片转3D 管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- 输入 / 输出 ---
    io_group = parser.add_argument_group("输入 / 输出")
    io_group.add_argument("--image", type=str, default=None,
                        help="输入图片路径 (默认: assets/example_image/ 中的第一张图片)")
    io_group.add_argument("--output-dir", type=str, default="out",
                        help="输出目录 (默认: out)")
    io_group.add_argument("--output-mesh", type=str, default=None,
                        help="输出 GLB 文件路径 (默认: {output-dir}/sample.glb)")
    io_group.add_argument("--output-video", type=str, default=None,
                        help="输出视频文件路径 (默认: {output-dir}/sample.mp4)")
    io_group.add_argument("--no-video", action="store_true",
                        help="跳过视频渲染")
    io_group.add_argument("--no-glb", action="store_true",
                        help="跳过 GLB 导出")
    io_group.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="pipeline.json 配置文件路径 (默认: TRELLIS.2-4B 的 pipeline.json)")

    # --- 管线 ---
    pipe_group = parser.add_argument_group("管线")
    pipe_group.add_argument("--seed", type=int, default=42,
                          help="随机种子 (默认: 42)")
    pipe_group.add_argument("--pipeline-type", type=str, default=None,
                          choices=['512', '1024', '1024_cascade', '1536_cascade'],
                          help="管线分辨率类型 (默认: 来自 pipeline.json)")
    pipe_group.add_argument("--preprocess-image", default=True, action=argparse.BooleanOptionalAction,
                          help="预处理图片（去除背景等）(默认: True)")
    pipe_group.add_argument("--max-num-tokens", type=int, default=49152,
                          help="最大 token 数量 (默认: 49152)")

    # --- 稀疏结构采样器 ---
    ss_group = parser.add_argument_group("稀疏结构采样器")
    ss_group.add_argument("--ss-steps", type=int, default=None,
                        help="稀疏结构采样步数 (默认: 来自 pipeline.json)")
    ss_group.add_argument("--ss-guidance-strength", type=float, default=None,
                        help="稀疏结构引导强度 (默认: 来自 pipeline.json)")
    ss_group.add_argument("--ss-guidance-rescale", type=float, default=None,
                        help="稀疏结构引导重缩放 (默认: 来自 pipeline.json)")
    ss_group.add_argument("--ss-rescale-t", type=float, default=None,
                        help="稀疏结构重缩放 T (默认: 来自 pipeline.json)")

    # --- 形状 SLat 采样器 ---
    shape_group = parser.add_argument_group("形状 SLat 采样器")
    shape_group.add_argument("--shape-steps", type=int, default=None,
                           help="形状 SLat 采样步数 (默认: 来自 pipeline.json)")
    shape_group.add_argument("--shape-guidance-strength", type=float, default=None,
                           help="形状 SLat 引导强度 (默认: 来自 pipeline.json)")
    shape_group.add_argument("--shape-guidance-rescale", type=float, default=None,
                           help="形状 SLat 引导重缩放 (默认: 来自 pipeline.json)")
    shape_group.add_argument("--shape-rescale-t", type=float, default=None,
                           help="形状 SLat 重缩放 T (默认: 来自 pipeline.json)")

    # --- 纹理 SLat 采样器 ---
    tex_group = parser.add_argument_group("纹理 SLat 采样器")
    tex_group.add_argument("--tex-steps", type=int, default=None,
                          help="纹理 SLat 采样步数 (默认: 来自 pipeline.json)")
    tex_group.add_argument("--tex-guidance-strength", type=float, default=None,
                          help="纹理 SLat 引导强度 (默认: 来自 pipeline.json)")
    tex_group.add_argument("--tex-guidance-rescale", type=float, default=None,
                          help="纹理 SLat 引导重缩放 (默认: 来自 pipeline.json)")
    tex_group.add_argument("--tex-rescale-t", type=float, default=None,
                          help="纹理 SLat 重缩放 T (默认: 来自 pipeline.json)")

    # --- GLB 导出 ---
    glb_group = parser.add_argument_group("GLB 导出")
    glb_group.add_argument("--decimation-target", type=int, default=1000000,
                         help="简化后的目标面数 (默认: 1000000)")
    glb_group.add_argument("--texture-size", type=int, default=4096,
                         help="输出纹理分辨率 (默认: 4096)")
    glb_group.add_argument("--remesh", default=True, action=argparse.BooleanOptionalAction,
                         help="重新网格化 (默认: True)")
    glb_group.add_argument("--remesh-band", type=int, default=1,
                         help="重网格化带宽 (默认: 1)")
    glb_group.add_argument("--remesh-project", type=int, default=0,
                         help="重网格化投影迭代次数 (默认: 0)")

    return parser


def build_sampler_params(args, cli_prefix: str, cfg_defaults: dict, param_names: list) -> dict:
    """Build a sampler params dict from CLI args, falling back to config defaults."""
    params = {}
    for name in param_names:
        cli_val = getattr(args, f'{cli_prefix}_{name}', None)
        if cli_val is not None:
            params[name] = cli_val
        elif name in cfg_defaults:
            params[name] = cfg_defaults[name]
    return params


def get_default_image() -> str:
    """Return the first image from assets/example_image/ or raise."""
    img_dir = "assets/example_image"
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if images:
        return os.path.join(img_dir, sorted(images)[0])
    raise FileNotFoundError(f"No images found in {img_dir}. Use --image to specify one.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Resolve input image ---
    image_path = args.image or get_default_image()
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)
    print(f"[1/5] Loading image: {image_path}")

    # --- Load config defaults ---
    config = {}
    if os.path.exists(args.config):
        print(f"[2/5] Loading config: {args.config}")
        config = load_config(args.config)
    else:
        print(f"[2/5] Config not found at {args.config}, using built-in defaults")

    # --- Resolve pipeline_type ---
    pipeline_type = args.pipeline_type or config.get('pipeline_type', '1024_cascade')

    # --- Build sampler params ---
    sampler_param_names = ['steps', 'guidance_strength', 'guidance_rescale', 'rescale_t']
    ss_params = build_sampler_params(args, 'ss', config.get('sparse_structure_sampler_params', {}), sampler_param_names)
    shape_params = build_sampler_params(args, 'shape', config.get('shape_slat_sampler_params', {}), sampler_param_names)
    tex_params = build_sampler_params(args, 'tex', config.get('tex_slat_sampler_params', {}), sampler_param_names)

    print(f"    Seed: {args.seed}")
    print(f"    Pipeline type: {pipeline_type}")
    print(f"    Sparse structure params: {ss_params}")
    print(f"    Shape SLat params: {shape_params}")
    if tex_params:
        print(f"    Texture SLat params: {tex_params}")
    else:
        print("    Texture SLat: disabled (no config or CLI params)")

    # --- Setup environment map ---
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    # --- Load Pipeline ---
    print("[3/5] Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # --- Load Image & Run ---
    image = Image.open(image_path)
    print("[4/5] Running pipeline (this may take a while)...")
    outputs = pipeline.run(
        image,
        seed=args.seed,
        preprocess_image=args.preprocess_image,
        pipeline_type=pipeline_type,
        max_num_tokens=args.max_num_tokens,
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
    )
    mesh = outputs[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    # --- Ensure output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_video = args.output_video or os.path.join(args.output_dir, "sample.mp4")
    output_mesh = args.output_mesh or os.path.join(args.output_dir, "sample.glb")

    # --- Render Video ---
    if not args.no_video:
        print("[5/5] Rendering video...")
        try:
            video = render_utils.make_pbr_vis_frames(
                render_utils.render_video(mesh, envmap=envmap)
            )
            imageio.mimsave(output_video, video, fps=15)
            print(f"    Video saved: {output_video}")
        except Exception as e:
            print(f"    Video rendering failed: {e}")

    # --- Export to GLB ---
    if not args.no_glb:
        print("    Exporting GLB...")
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=args.remesh,
                remesh_band=args.remesh_band,
                remesh_project=args.remesh_project,
                verbose=True,
            )
            glb.export(output_mesh, extension_webp=True)
            print(f"    GLB saved: {output_mesh}")
        except Exception as e:
            print(f"    GLB export failed: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
