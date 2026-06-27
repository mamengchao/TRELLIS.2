"""
TRELLIS.2 纹理生成命令行工具
=============================
使用示例:
    python example_texturing.py --mesh assets/example_texturing/the_forgotten_knight.ply --image assets/example_texturing/image.webp
    python example_texturing.py --mesh out/model.glb --tex-steps 20 --resolution 1024
"""
import os
import argparse
import json
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_config(config_path: str) -> dict:
    """加载 texturing_pipeline.json，提取默认采样器参数。"""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    args = cfg.get('args', {})
    defaults = {
        'tex_slat_sampler_params': args.get('tex_slat_sampler', {}).get('params', {}),
    }
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 纹理生成管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- 输入 / 输出 ---
    io_group = parser.add_argument_group("输入 / 输出")
    io_group.add_argument("--mesh", type=str, required=True,
                        help="输入网格文件路径（支持 .ply、.glb、.obj 等格式）")
    io_group.add_argument("--image", type=str, required=True,
                        help="输入纹理参考图片路径")
    io_group.add_argument("--output", type=str, default=None,
                        help="输出 GLB 文件路径（默认: 输入网格所在目录/<网格名>_textured.glb）")
    io_group.add_argument("--config", type=str, default=None,
                        help="texturing_pipeline.json 配置文件路径（不指定则使用 HuggingFace 模型自带的配置）")

    # --- 管线 ---
    pipe_group = parser.add_argument_group("管线")
    pipe_group.add_argument("--seed", type=int, default=42,
                          help="随机种子（默认: 42）")
    pipe_group.add_argument("--resolution", type=int, default=1024,
                          choices=[512, 1024],
                          help="纹理分辨率（512 或 1024，默认: 1024）")
    pipe_group.add_argument("--texture-size", type=int, default=2048,
                          help="输出纹理贴图尺寸（默认: 2048）")
    pipe_group.add_argument("--preprocess-image", default=True,
                          action=argparse.BooleanOptionalAction,
                          help="预处理图片（去除背景等）（默认: True）")

    # --- 纹理 SLat 采样器 ---
    tex_group = parser.add_argument_group("纹理 SLat 采样器")
    tex_group.add_argument("--tex-steps", type=int, default=None,
                          help="采样步数（默认: 12）")
    tex_group.add_argument("--tex-guidance-strength", type=float, default=None,
                          help="引导强度（默认: 1.0）")
    tex_group.add_argument("--tex-guidance-rescale", type=float, default=None,
                          help="引导重缩放（默认: 0.0）")
    tex_group.add_argument("--tex-rescale-t", type=float, default=None,
                          help="重缩放 T（默认: 3.0）")

    return parser


def build_sampler_params(args, cli_prefix: str, cfg_defaults: dict, param_names: list) -> dict:
    """从 CLI 参数构建采样器参数字典，未指定的值回退到配置文件默认值。"""
    params = {}
    for name in param_names:
        cli_val = getattr(args, f'{cli_prefix}_{name}', None)
        if cli_val is not None:
            params[name] = cli_val
        elif name in cfg_defaults:
            params[name] = cfg_defaults[name]
    return params


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- 检查输入文件 ---
    if not os.path.exists(args.mesh):
        print(f"错误: 网格文件不存在: {args.mesh}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在: {args.image}")
        sys.exit(1)

    # --- 延迟加载重型库（加速 --help 响应）---
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2TexturingPipeline

    print(f"[1/4] 加载网格: {args.mesh}")
    mesh = trimesh.load(args.mesh, force="mesh")
    print(f"      顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")

    print(f"[2/4] 加载图片: {args.image}")
    image = Image.open(args.image)

    # --- 加载配置默认值（可选） ---
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            sys.exit(1)
        print(f"[3/4] 加载配置文件: {args.config}")
        config = load_config(args.config)
    else:
        print("[3/4] 使用 HuggingFace 模型自带的配置")

    # --- 构建采样器参数 ---
    sampler_param_names = ['steps', 'guidance_strength', 'guidance_rescale', 'rescale_t']
    tex_params = build_sampler_params(args, 'tex', config.get('tex_slat_sampler_params', {}), sampler_param_names)

    print(f"      随机种子: {args.seed}")
    print(f"      分辨率: {args.resolution}")
    print(f"      纹理尺寸: {args.texture_size}")
    print(f"      纹理采样器参数: {tex_params}")

    # --- 加载管线 ---
    print("      加载纹理生成管线...")
    pipeline = Trellis2TexturingPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json"
    )
    pipeline.cuda()

    # --- 运行 ---
    print("      运行纹理生成管线（这可能需要一些时间）...")
    result = pipeline.run(
        mesh,
        image,
        seed=args.seed,
        preprocess_image=args.preprocess_image,
        resolution=args.resolution,
        texture_size=args.texture_size,
        tex_slat_sampler_params=tex_params,
    )

    # --- 导出 ---
    if args.output is None:
        mesh_dir = os.path.dirname(args.mesh) or "."
        mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output = os.path.join(mesh_dir, f"{mesh_name}_textured.glb")

    print(f"[4/4] 导出结果: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.export(args.output, extension_webp=True)
    print("      完成！")


if __name__ == "__main__":
    main()
