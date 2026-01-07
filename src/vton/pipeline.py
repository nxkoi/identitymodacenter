"""
Módulo VTON (Virtual Try-On)
Integra Stable Diffusion 1.5 com gerenciamento de identidade e guarda-roupa
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
import numpy as np

from ..core import GPUManager
from ..identity import IdentityManager
from ..wardrobe import WardrobeManager

logger = logging.getLogger(__name__)


class VTONPipeline:
    """
    Pipeline de Virtual Try-On
    Otimizado para RTX 4070 (8GB VRAM)
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        identity_manager: IdentityManager,
        wardrobe_manager: WardrobeManager,
        config: Dict[str, Any]
    ):
        """
        Inicializa o pipeline VTON
        
        Args:
            gpu_manager: Gerenciador de GPU
            identity_manager: Gerenciador de identidade
            wardrobe_manager: Gerenciador de guarda-roupa
            config: Configuração do sistema
        """
        self.gpu_manager = gpu_manager
        self.identity_manager = identity_manager
        self.wardrobe_manager = wardrobe_manager
        self.config = config
        
        # Stable Diffusion será carregado sob demanda
        self.sd_pipeline = None
        
        # Configurações VTON
        vton_config = config.get('vton', {})
        self.output_path = Path(vton_config.get('output_path', 'outputs'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.image_size = tuple(vton_config.get('image_size', [512, 768]))
        self.num_inference_steps = vton_config.get('num_inference_steps', 30)
        self.guidance_scale = vton_config.get('guidance_scale', 7.5)
        
        logger.info("VTONPipeline inicializado")
    
    def _load_stable_diffusion(self):
        """
        Carrega o modelo Stable Diffusion 1.5
        IMPORTANTE: Nunca carregar junto com InsightFace
        """
        if self.sd_pipeline is not None:
            return
        
        try:
            # Garantir que InsightFace não está carregado
            self.gpu_manager.ensure_exclusive_loading('sd15', ['insightface'])
            
            # Descarregar InsightFace se estiver na memória
            self.identity_manager.unload_model()
            
            logger.info("Carregando Stable Diffusion 1.5...")
            
            # Importar aqui para evitar importação desnecessária
            from diffusers import StableDiffusionInpaintPipeline
            
            # Configuração do SD
            model_config = self.config.get('models', {}).get('stable_diffusion', {})
            model_id = model_config.get('model_id', 'runwayml/stable-diffusion-v1-5')
            variant = model_config.get('variant', 'fp16')
            
            # Carregar pipeline
            self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=self.gpu_manager.dtype,
                variant=variant if self.gpu_manager.use_float16 else None,
                safety_checker=None if not model_config.get('safety_checker', False) else 'default'
            )
            
            # Otimizações para baixa VRAM
            if self.gpu_manager.cpu_offload:
                logger.info("Ativando CPU offload para Stable Diffusion")
                self.sd_pipeline.enable_model_cpu_offload()
            else:
                self.sd_pipeline = self.sd_pipeline.to(self.gpu_manager.device)
            
            # Ativar attention slicing para economizar memória
            self.sd_pipeline.enable_attention_slicing(1)
            
            # Ativar VAE slicing
            if hasattr(self.sd_pipeline, 'enable_vae_slicing'):
                self.sd_pipeline.enable_vae_slicing()
            
            logger.info("Stable Diffusion 1.5 carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar Stable Diffusion: {e}")
            raise
    
    def _prepare_images(
        self,
        person_image: Image.Image,
        garment_image: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Prepara as imagens para o processo de VTON
        
        Args:
            person_image: Imagem da pessoa
            garment_image: Imagem da peça de roupa
        
        Returns:
            Tupla de (imagem_pessoa_redimensionada, imagem_roupa_redimensionada, máscara)
        """
        # Redimensionar imagens
        person_image = person_image.resize(self.image_size, Image.LANCZOS)
        garment_image = garment_image.resize(self.image_size, Image.LANCZOS)
        
        # Criar máscara simples (região do torso)
        # Em uma implementação real, você usaria um modelo de segmentação
        mask = Image.new('L', self.image_size, 255)
        
        # Região aproximada do torso (centro da imagem)
        width, height = self.image_size
        torso_top = int(height * 0.25)
        torso_bottom = int(height * 0.75)
        torso_left = int(width * 0.25)
        torso_right = int(width * 0.75)
        
        # Criar máscara retangular
        mask_array = np.array(mask)
        mask_array[torso_top:torso_bottom, torso_left:torso_right] = 255
        mask = Image.fromarray(mask_array)
        
        return person_image, garment_image, mask
    
    def try_on(
        self,
        person_image: Image.Image,
        garment_item_id: str,
        prompt: str = "person wearing clothing, high quality, detailed",
        negative_prompt: str = "blurry, low quality, distorted",
        save_output: bool = True
    ) -> Optional[Image.Image]:
        """
        Realiza o try-on virtual
        
        Args:
            person_image: Imagem da pessoa
            garment_item_id: ID do item de roupa do guarda-roupa
            prompt: Prompt para geração
            negative_prompt: Prompt negativo
            save_output: Salvar resultado em disco
        
        Returns:
            Imagem resultante ou None em caso de erro
        """
        try:
            # Carregar imagem da roupa
            garment_image = self.wardrobe_manager.get_item_image(garment_item_id)
            if garment_image is None:
                logger.error(f"Item {garment_item_id} não encontrado")
                return None
            
            # Preparar imagens
            person_image, garment_image, mask = self._prepare_images(person_image, garment_image)
            
            # Carregar Stable Diffusion
            if self.sd_pipeline is None:
                self._load_stable_diffusion()
            
            logger.info(f"Iniciando try-on com item {garment_item_id}")
            
            with self.gpu_manager.safe_execution():
                # Gerar imagem com inpainting
                with self.gpu_manager.model_context('sd15', self.sd_pipeline):
                    result = self.sd_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=person_image,
                        mask_image=mask,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        height=self.image_size[1],
                        width=self.image_size[0]
                    )
                    
                    output_image = result.images[0]
            
            # Salvar resultado
            if save_output:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tryon_{garment_item_id}_{timestamp}.png"
                output_path = self.output_path / output_filename
                output_image.save(output_path)
                logger.info(f"Resultado salvo em {output_path}")
            
            logger.info("Try-on concluído com sucesso")
            return output_image
            
        except MemoryError as e:
            logger.error(f"Erro de memória durante try-on: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro durante try-on: {e}")
            return None
    
    def try_on_with_identity(
        self,
        identity_id: str,
        garment_item_id: str,
        base_image: Image.Image,
        prompt: Optional[str] = None,
        save_output: bool = True
    ) -> Optional[Image.Image]:
        """
        Realiza try-on usando identidade registrada
        
        Args:
            identity_id: ID da identidade
            garment_item_id: ID do item de roupa
            base_image: Imagem base da pessoa
            prompt: Prompt customizado (None para usar padrão)
            save_output: Salvar resultado
        
        Returns:
            Imagem resultante ou None
        """
        # Verificar se identidade existe
        identity_embedding = self.identity_manager.get_identity_embedding(identity_id)
        if identity_embedding is None:
            logger.error(f"Identidade {identity_id} não encontrada")
            return None
        
        # Prompt padrão se não fornecido
        if prompt is None:
            prompt = f"person with identity {identity_id} wearing clothing, high quality, detailed, realistic"
        
        logger.info(f"Try-on com identidade {identity_id}")
        
        return self.try_on(
            person_image=base_image,
            garment_item_id=garment_item_id,
            prompt=prompt,
            save_output=save_output
        )
    
    def unload_model(self):
        """Descarrega o modelo Stable Diffusion da memória"""
        if self.sd_pipeline is not None:
            logger.info("Descarregando Stable Diffusion")
            self.sd_pipeline = None
            self.gpu_manager.clear_memory()
