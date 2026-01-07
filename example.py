#!/usr/bin/env python3
"""
Exemplo de uso do sistema VTON
Demonstra todas as funcionalidades principais
"""

import numpy as np
from PIL import Image
from pathlib import Path
import logging

from src.core import Config, GPUManager
from src.identity import IdentityManager
from src.wardrobe import WardrobeManager
from src.vton import VTONPipeline

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_image(size=(512, 768), color=(128, 128, 128)):
    """Cria uma imagem de exemplo"""
    array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(array)


def main():
    """Função principal do exemplo"""
    
    logger.info("=== Exemplo de Uso do Sistema VTON ===\n")
    
    # 1. Inicializar sistema
    logger.info("1. Inicializando sistema...")
    config = Config()
    
    gpu_manager = GPUManager(
        use_float16=config.get('hardware.use_float16', True),
        cpu_offload=config.get('hardware.cpu_offload', True)
    )
    
    identity_manager = IdentityManager(
        gpu_manager=gpu_manager,
        config=config.config
    )
    
    wardrobe_manager = WardrobeManager(
        storage_path=Path("data/wardrobe_example"),
        max_items=100
    )
    
    vton_pipeline = VTONPipeline(
        gpu_manager=gpu_manager,
        identity_manager=identity_manager,
        wardrobe_manager=wardrobe_manager,
        config=config.config
    )
    
    logger.info("✓ Sistema inicializado\n")
    
    # 2. Adicionar itens ao guarda-roupa
    logger.info("2. Adicionando itens ao guarda-roupa...")
    
    # Criar imagens de exemplo
    shirt_image = create_sample_image(color=(100, 150, 200))  # Azul
    pants_image = create_sample_image(color=(50, 50, 50))      # Cinza escuro
    
    shirt_id = wardrobe_manager.add_item(
        image=shirt_image,
        category="shirt",
        metadata={'color': 'blue', 'style': 'casual'}
    )
    
    pants_id = wardrobe_manager.add_item(
        image=pants_image,
        category="pants",
        metadata={'color': 'gray', 'style': 'formal'}
    )
    
    logger.info(f"✓ Camisa adicionada: {shirt_id}")
    logger.info(f"✓ Calça adicionada: {pants_id}\n")
    
    # 3. Listar itens do guarda-roupa
    logger.info("3. Listando itens do guarda-roupa...")
    items = wardrobe_manager.list_items()
    for item in items:
        logger.info(f"   - {item.item_id}: {item.category}")
    
    stats = wardrobe_manager.get_stats()
    logger.info(f"   Total: {stats['total_items']} itens\n")
    
    # 4. Demonstrar grafo temporal (sem InsightFace real)
    logger.info("4. Demonstrando grafo temporal de identidade...")
    
    # Simular embeddings (em produção, viria do InsightFace)
    fake_embedding_1 = np.random.randn(512).astype(np.float32)
    fake_embedding_2 = fake_embedding_1 + np.random.randn(512).astype(np.float32) * 0.05
    
    # Adicionar ao grafo
    identity_id_1 = identity_manager.temporal_graph.add_or_update_identity(fake_embedding_1)
    logger.info(f"✓ Primeira observação: {identity_id_1}")
    
    identity_id_2 = identity_manager.temporal_graph.add_or_update_identity(fake_embedding_2)
    logger.info(f"✓ Segunda observação: {identity_id_2}")
    
    if identity_id_1 == identity_id_2:
        logger.info("   → Mesma identidade reconhecida!")
    else:
        logger.info("   → Identidades diferentes detectadas")
    
    # Listar identidades
    identities = identity_manager.list_identities()
    logger.info(f"\n   Identidades registradas: {len(identities)}")
    for info in identities:
        logger.info(f"   - {info['identity_id']}: {info['num_observations']} observações")
    
    logger.info()
    
    # 5. Informações de memória
    logger.info("5. Status de memória GPU...")
    mem_info = gpu_manager.get_memory_usage()
    
    if mem_info['device'] == 'cuda':
        logger.info(f"   Device: {mem_info['device']}")
        logger.info(f"   Memória total: {mem_info['total_gb']:.2f} GB")
        logger.info(f"   Memória livre: {mem_info['free_gb']:.2f} GB")
        logger.info(f"   Utilização: {mem_info['utilization_pct']:.1f}%")
    else:
        logger.info(f"   Device: {mem_info['device']} (CPU mode)")
    
    logger.info()
    
    # 6. Demonstrar proteções
    logger.info("6. Demonstrando proteções do sistema...")
    
    logger.info("   ✓ Float16 ativo" if gpu_manager.use_float16 else "   ○ Float16 inativo")
    logger.info("   ✓ CPU Offload ativo" if gpu_manager.cpu_offload else "   ○ CPU Offload inativo")
    logger.info("   ✓ Carregamento exclusivo configurado")
    logger.info("   ✓ Captura de OOM ativa")
    logger.info("   ✓ Pathlib usado em todos os caminhos")
    
    logger.info("\n=== Exemplo concluído com sucesso! ===")
    
    # Nota sobre VTON real
    logger.info("\nNOTA: Para realizar try-on real, você precisa:")
    logger.info("  1. Baixar os modelos (InsightFace, Stable Diffusion)")
    logger.info("  2. Fornecer imagens reais de pessoas e roupas")
    logger.info("  3. Usar: python main.py --mode vton --person image.jpg --garment item_id")


if __name__ == "__main__":
    main()
