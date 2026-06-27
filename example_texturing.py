import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

# 1. Load Pipeline
pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

# 2. Load Mesh, image & Run
# mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
mesh = trimesh.load_mesh("assets/example_texturing/the_forgotten_knight.ply","ply")
image = Image.open("assets/example_texturing/image.webp")

# mesh = trimesh.load_mesh("out/sample.glb",'glb')
# image = Image.open("assets/example_image/8ce83f6a28910e755902de10918672e77dd23476f43f0f1521c48667de6cea84.webp")
output = pipeline.run(mesh, image)

# 3. Render Mesh
output.export("out/the_forgotten_knight_textured.glb", extension_webp=True)
