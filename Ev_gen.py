# Instalación de dependencias
#!git clone https://github.com/NVlabs/stylegan3.git
# %cd stylegan3

#!pip install torch torchvision lpips transformers pillow numpy tqdm click ninja gradio
#!pip install torchvision

#import sys
#sys.path.append('/content/stylegan3')

# Descargar el modelo preentrenado
#!wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl -O /content/model.pkl

# Importaciones necesarias
import os
import re
from typing import List, Optional
from tqdm import tqdm
import dnnlib
import numpy as np
import PIL.Image
import torch
import lpips
from transformers import CLIPModel
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
#from google.colab import drive
import legacy
import gradio as gr
import requests
from io import BytesIO
import tempfile

# Configuración inicial
#drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Cargar modelos
print(f'Loading models on {device}...')
lpips_model = lpips.LPIPS(net='alex').to(device).eval()
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()

# Función para cargar imágenes de referencia con manejo de errores
def load_and_process_ref_images(ref_images):
    ref_images_pil = []
    for img in ref_images:
        try:
            # Si es un archivo subido (tiene atributo 'name')
            if hasattr(img, 'name'):
                img_pil = Image.open(img.name).convert('RGB')
            # Si es una ruta de archivo (string)
            elif isinstance(img, str) and os.path.exists(img):
                img_pil = Image.open(img).convert('RGB')
            # Si es una URL (string que comienza con http)
            elif isinstance(img, str) and img.startswith('http'):
                response = requests.get(img)
                img_pil = Image.open(BytesIO(response.content)).convert('RGB')
            # Si es un array numpy
            elif isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img).convert('RGB')
            else:
                print(f"Formato de imagen no reconocido: {type(img)}")
                continue

            ref_images_pil.append(img_pil)
        except Exception as e:
            print(f"Error al procesar imagen {img}: {str(e)}")
            continue

    if not ref_images_pil:
        raise ValueError("No se pudieron cargar imágenes de referencia válidas")

    # Preprocesamiento para LPIPS
    ref_lpips_tensors = torch.stack([TF.to_tensor(TF.resize(img, (256, 256))) * 2 -1 for img in ref_images_pil]).to(device)

    # Preprocesamiento para CLIP
    clip_norm = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ref_clip_input = torch.stack([clip_norm(TF.to_tensor(TF.resize(img, (224, 224)))) for img in ref_images_pil]).to(device)

    with torch.no_grad():
        ref_clip_features = clip_model.get_image_features(ref_clip_input)
        ref_clip_features /= ref_clip_features.norm(dim=-1, keepdim=True)

    return ref_lpips_tensors, ref_clip_features

# Función de evaluación de fitness
def evaluate_fitness(population_2D, generator, ref_lpips_tensors, ref_clip_features, alpha=0.5, beta=0.5):
    """Vectorized function to evaluate batch fitness from tensors"""
    with torch.no_grad():
        # 1. Reshape y generar batch de imágenes
        pop_3d = population_2D.reshape(population_2D.shape[0], 16, 512)
        w_tensor = torch.from_numpy(pop_3d).float().to(device)
        generated_batch = generator.synthesis(w_tensor, noise_mode='const')

        # 2. Calculo de LPIPS
        generated_lpips_input = TF.resize(generated_batch, (256, 256))

        # Asegurar que las dimensiones coincidan para LPIPS
        # Si hay múltiples imágenes de referencia, calcular el promedio
        lpips_scores = []
        for i in range(generated_lpips_input.shape[0]):
            dist_lpips = lpips_model(generated_lpips_input[i:i+1].repeat(ref_lpips_tensors.shape[0], 1, 1, 1), ref_lpips_tensors)
            lpips_scores.append(dist_lpips.mean().item())
        lpips_scores = torch.tensor(lpips_scores, device=device)

        # 3. Calculo de CLIP
        generated_clip_input = TF.resize(generated_batch, (224, 224))
        clip_norm = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        generated_clip_input = clip_norm((generated_clip_input + 1)/2)

        gen_clip_features = clip_model.get_image_features(generated_clip_input)
        gen_clip_features /= gen_clip_features.norm(dim=-1, keepdim=True)

        # Calcular similitud coseno con todas las imágenes de referencia y promediar
        clip_similarities = torch.matmul(gen_clip_features, ref_clip_features.T)
        clip_scores = 1 - clip_similarities.mean(dim=1)

        # Fitness combinado
        Fitness_total = (alpha * lpips_scores + beta * clip_scores)

    return Fitness_total.cpu().numpy()

# Función para generar imágenes
def generate_img(w_optimized, generator, noise_mode):
    """Generar imágenes basadas en los espacios latentes optimizados"""
    with torch.no_grad():
        pop_3d = w_optimized.reshape(1, 16, 512)
        w_tensor = torch.from_numpy(pop_3d).float().to(device)
        best_image = generator.synthesis(w_tensor, noise_mode=noise_mode)
        imagen = (best_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return imagen[0].cpu().numpy()

# Función de optimización adaptada para Gradio
def optimize_latent_space(
    network_pkl: str,
    truncation_psi: float = 1.0,
    outdir: str = './out_hmi',
    noise_mode: str = 'const',
    images: int = 2,  # Cambiado a 2 para generar dos imágenes
    poblacion: int = 20,
    generations: int = 400,
    tau1:float = 0.1,
    tau2:float = 0.1,
    ref_lpips_tensors=None,
    ref_clip_features=None
):
    """Optimizador de espacios latentes W basado en jDE"""

    ndim = 8192                 # Dimensión del problema
    F_min, F_max = 0.1, 0.9     # Rango de F mutacion
    #tau1, tau2 = 0.1, 0.3       # Probabilidad de cambiar F y CR
    bounds = (-3, 3)            # Límites de búsqueda
    F = np.random.uniform(F_min, F_max, size=poblacion)
    CR = np.random.uniform(0, 1, size=poblacion)

    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)

    best_vectors = []

    for _ in range(images):
        # 1. Generar el vector Z y el vector W inicial
        batch_z = torch.randn(poblacion, G.z_dim, device=device)
        batch_labels = label.repeat(poblacion, 1)
        w_batch = G.mapping(batch_z, batch_labels, truncation_psi=truncation_psi)

        np_3Dmatrix = w_batch.detach().cpu().numpy()
        assert np_3Dmatrix.shape == (poblacion, 16, 512), "La forma de la matriz 3D es incorrecta"
        np_2Dvector = np_3Dmatrix.reshape(poblacion, -1)

        # jDE Loop
        print("Calculando fitness inicial...")
        fitness = evaluate_fitness(np_2Dvector, G, ref_lpips_tensors, ref_clip_features)
        print(f'Fitness inicial (mejor):{fitness.min():.5f}')

        for gen in tqdm(range(generations), desc='Generating images'):
            # 1. Auto-adaptar F y CR (Vectorizado)
            F_rand = np.random.rand(poblacion)
            CR_rand = np.random.rand(poblacion)
            F[F_rand < tau1] = F_min + np.random.rand(np.sum(F_rand < tau1)) * (F_max - F_min)
            CR[CR_rand < tau2] = np.random.rand(np.sum(CR_rand < tau2))

            # Generacion de poblacion de prueba
            idx = np.arange(poblacion)
            choices = np.array([np.random.choice(np.delete(idx, i), 3, replace=False) for i in range(poblacion)])
            r1, r2, r3 = np_2Dvector[choices[:, 0]], np_2Dvector[choices[:, 1]], np_2Dvector[choices[:, 2]]

            # Mutacion y crossover
            mutant = r1 + F[:,np.newaxis] * (r2 - r3)
            mutant = np.clip(mutant, bounds[0], bounds[1])

            cross_points = np.random.rand(poblacion, ndim) < CR[:, np.newaxis]
            rand_idx = np.random.randint(0, ndim, size=poblacion)
            cross_points[idx,rand_idx] = True
            trial_population = np.where(cross_points, mutant, np_2Dvector)

            f_trial = evaluate_fitness(trial_population, G, ref_lpips_tensors, ref_clip_features)

            # Seleccion
            mejora = f_trial < fitness
            np_2Dvector[mejora] = trial_population[mejora]
            fitness[mejora] = f_trial[mejora]

            if gen % 10 == 0 or gen == generations - 1:
                print(f"Gen {gen+1}/{generations}: Best fitness = {fitness.min():.5f}")

        # Resultado final
        best_idx = np.argmin(fitness)
        best_vector = np_2Dvector[best_idx]
        best_vectors.append(best_vector)
        print("Optimización finalizada")
        print(f"Mejor fitness: {fitness.min():.5f}")

    return best_vectors  # Siempre devolver todas las imágenes

# Función principal para Gradio con manejo de errores
def generate_images_with_gradio(ref_images, truncation_psi, noise_mode, population_size, generations, tau1, tau2):
    try:
        if not ref_images or len(ref_images) == 0:
            raise ValueError("Por favor sube al menos una imagen de referencia")

        # Mostrar progreso
        progress = gr.Progress()
        progress(0, desc="Procesando imágenes de referencia...")

        # Procesar imágenes de referencia
        ref_lpips_tensors, ref_clip_features = load_and_process_ref_images(ref_images)
        print(f"Procesadas {ref_lpips_tensors.shape[0]} imágenes de referencia")

        # Cargar modelo
        network_pkl = "stylegan2-ada-pytorch/models/afhqcat.pkl"

        # Ejecutar optimización
        progress(0.3, desc="Optimizando espacio latente...")
        w_optimized_list = optimize_latent_space(
            network_pkl=network_pkl,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            outdir='./out_hmi',
            images=2,  # Generar dos imágenes
            poblacion=population_size,
            generations=generations,
            tau1 = tau1,
            tau2 =tau2,
            ref_lpips_tensors=ref_lpips_tensors,
            ref_clip_features=ref_clip_features
        )

        # Generar imágenes finales
        progress(0.8, desc="Generando imágenes finales...")
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)

        generated_images = []
        for i, w_optimized in enumerate(w_optimized_list):
            img = generate_img(w_optimized, G, noise_mode)
            generated_images.append(img)

        progress(1.0, desc="¡Imágenes generadas con éxito!")
        return generated_images, "Generación completada con éxito ✅"  # Imágenes + mensaje

    except Exception as e:
        print(f"Error durante la generación: {str(e)}")
        error_msg = f"Error: {str(e)}"
        if "CUDA out of memory" in str(e):
            error_msg = "\n\nLimite de memoria alcanzado, prueba reducir el tamaño de población o generaciones."
        return None, error_msg

# Crear interfaz Gradio mejorada # 🎨 StyleGAN3 Image Generator
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # E-GAN
        ### Genera 2 imágenes basadas en tus referencias usando IA generativa
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.File(
                    label="Imágenes de referencia",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    height=150
                )

                with gr.Accordion("⚙️ Parámetros avanzados", open=False):
                    truncation = gr.Slider(
                        minimum=0.1, maximum=2.0, step=0.1,
                        value=1.0, label="Truncation Psi (controla la variación)"
                    )
                    noise_mode = gr.Dropdown(
                        ["const", "random", "none"],
                        value="const",
                        label="Modo de Ruido"
                    )
                    population_size = gr.Slider(
                        minimum=5, maximum=30, step=5,
                        value=20,
                        label="Tamaño de población (mayor = más lento pero mejor calidad)"
                    )
                    generations = gr.Slider(
                        minimum=10, maximum=600, step=10,
                        value=100,
                        label="Generaciones (mayor = más lento pero mejor calidad)"
                    )
                    tau1 = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.1,
                        value=0.1,
                        label="Probabilidad de cambio de F (mutacion)"
                    )
                    tau2 = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.1,
                        value=0.1,
                        label="Probabilidad de cambio de CR (cruce)"
                    )
                generate_btn = gr.Button("✨ Generar Imágenes", variant="primary")

                # Sección de ejemplos con URLs
                with gr.Accordion("📸 Ejemplo para probar", open=True):
                    gr.Markdown("""
                    **URL de ejemplo:**
                    - Gato: https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&w=300
                    Copia y pega este URL en un cuadro de texto separado si quieres usarla como referencia.
                    """)

            with gr.Column():
                # Mostrar galería de imágenes generadas
                output_gallery = gr.Gallery(
                    label="Imágenes generadas",
                    columns=2,
                    height="auto"
                )
                status = gr.Textbox(label="Estado", interactive=False)

        # Manejo del botón con feedback visual
        generate_btn.click(
            fn=generate_images_with_gradio,
            inputs=[input_image, truncation, noise_mode, population_size, generations, tau1, tau2],
            outputs=[output_gallery, status],
            api_name="generate"
        )

    return demo

# Función para manejar mejor los errores al lanzar la interfaz
def safe_launch():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\nPreparando la interfaz (Intento {attempt + 1}/{max_retries})...")
            demo = create_interface()

            # Primero intentamos con share=False
            print("Intentando con share=False...")
            demo.launch(
                share=True,
                server_name="127.0.0.1",
                server_port=7861,
                show_error=True,
                debug=True
            )
            break

        except Exception as e:
            print(f"Error en intento {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                print("\n⚠️ No se pudo lanzar la interfaz después de varios intentos.")
                print("Posibles soluciones:")
                print("1. Reinicia el entorno de ejecución (Runtime -> Restart runtime)")
                print("2. Verifica tu conexión a internet")
                print("3. Intenta con un entorno local si persisten los problemas")
            else:
                print("Reintentando...")
                continue

# Lanzar la interfaz
if __name__ == "__main__":
    safe_launch()