#!/usr/bin/env python3
"""
Sistema VTON Modular Local
Otimizado para RTX 4070 (8GB VRAM)

Uso:
    python main.py --mode identity --action register --image path/to/image.jpg
    python main.py --mode wardrobe --action add --image path/to/garment.jpg --category shirt
    python main.py --mode vton --person path/to/person.jpg --garment item_0001
"""

import argparse
import logging
from pathlib import Path
from PIL import Image

from src.core import Config, GPUManager
from src.identity import IdentityManager
from src.wardrobe import WardrobeManager
from src.vton import VTONPipeline

logger = logging.getLogger(__name__)


def setup_system(config_path: Path = None) -> tuple:
    """
    Inicializa o sistema VTON completo
    
    Args:
        config_path: Caminho para arquivo de configuração
    
    Returns:
        Tupla (config, gpu_manager, identity_manager, wardrobe_manager, vton_pipeline)
    """
    # Carregar configuração
    config = Config(config_path)
    logger.info("=== Sistema VTON Modular Inicializado ===")
    
    # Configurar GPU Manager
    hardware_config = config.get('hardware', {})
    gpu_manager = GPUManager(
        use_float16=hardware_config.get('use_float16', True),
        cpu_offload=hardware_config.get('cpu_offload', True)
    )
    
    # Configurar Identity Manager
    identity_manager = IdentityManager(
        gpu_manager=gpu_manager,
        config=config.config
    )
    
    # Configurar Wardrobe Manager
    wardrobe_config = config.get('wardrobe', {})
    wardrobe_manager = WardrobeManager(
        storage_path=Path(wardrobe_config.get('storage_path', 'data/wardrobe')),
        max_items=wardrobe_config.get('max_items', 1000)
    )
    
    # Configurar VTON Pipeline
    vton_pipeline = VTONPipeline(
        gpu_manager=gpu_manager,
        identity_manager=identity_manager,
        wardrobe_manager=wardrobe_manager,
        config=config.config
    )
    
    return config, gpu_manager, identity_manager, wardrobe_manager, vton_pipeline


def mode_identity(args, identity_manager: IdentityManager):
    """Modo de gerenciamento de identidade"""
    
    if args.action == "register":
        if not args.image:
            logger.error("Necessário fornecer --image para registrar identidade")
            return
        
        image = Image.open(args.image)
        identity_id = identity_manager.register_identity(image, args.identity_id)
        
        if identity_id:
            logger.info(f"✓ Identidade registrada: {identity_id}")
        else:
            logger.error("✗ Falha ao registrar identidade")
    
    elif args.action == "identify":
        if not args.image:
            logger.error("Necessário fornecer --image para identificar pessoa")
            return
        
        image = Image.open(args.image)
        identity_id = identity_manager.identify_person(image)
        
        if identity_id:
            logger.info(f"✓ Pessoa identificada: {identity_id}")
        else:
            logger.error("✗ Nenhuma identidade reconhecida")
    
    elif args.action == "list":
        identities = identity_manager.list_identities()
        logger.info(f"\n=== Identidades Registradas ({len(identities)}) ===")
        for info in identities:
            logger.info(f"  - {info['identity_id']}: {info['num_observations']} observações")
    
    else:
        logger.error(f"Ação desconhecida: {args.action}")


def mode_wardrobe(args, wardrobe_manager: WardrobeManager):
    """Modo de gerenciamento de guarda-roupa"""
    
    if args.action == "add":
        if not args.image:
            logger.error("Necessário fornecer --image para adicionar item")
            return
        
        image = Image.open(args.image)
        item_id = wardrobe_manager.add_item(
            image=image,
            category=args.category or "clothing",
            metadata={'source': str(args.image)}
        )
        
        logger.info(f"✓ Item adicionado ao guarda-roupa: {item_id}")
    
    elif args.action == "list":
        items = wardrobe_manager.list_items(category=args.category)
        logger.info(f"\n=== Itens no Guarda-roupa ({len(items)}) ===")
        for item in items:
            logger.info(f"  - {item.item_id}: {item.category}")
    
    elif args.action == "stats":
        stats = wardrobe_manager.get_stats()
        logger.info(f"\n=== Estatísticas do Guarda-roupa ===")
        logger.info(f"  Total de itens: {stats['total_items']}/{stats['max_items']}")
        logger.info(f"  Categorias:")
        for category, count in stats['categories'].items():
            logger.info(f"    - {category}: {count}")
    
    elif args.action == "remove":
        if not args.item_id:
            logger.error("Necessário fornecer --item-id para remover item")
            return
        
        if wardrobe_manager.remove_item(args.item_id):
            logger.info(f"✓ Item removido: {args.item_id}")
        else:
            logger.error(f"✗ Item não encontrado: {args.item_id}")
    
    else:
        logger.error(f"Ação desconhecida: {args.action}")


def mode_vton(args, vton_pipeline: VTONPipeline):
    """Modo VTON (Virtual Try-On)"""
    
    if not args.person or not args.garment:
        logger.error("Necessário fornecer --person (imagem) e --garment (item_id)")
        return
    
    # Carregar imagem da pessoa
    person_image = Image.open(args.person)
    
    # Realizar try-on
    if args.identity_id:
        # Try-on com identidade registrada
        result = vton_pipeline.try_on_with_identity(
            identity_id=args.identity_id,
            garment_item_id=args.garment,
            base_image=person_image,
            prompt=args.prompt,
            save_output=True
        )
    else:
        # Try-on simples
        result = vton_pipeline.try_on(
            person_image=person_image,
            garment_item_id=args.garment,
            prompt=args.prompt or "person wearing clothing, high quality, detailed",
            save_output=True
        )
    
    if result:
        logger.info("✓ Try-on concluído com sucesso!")
    else:
        logger.error("✗ Falha no try-on")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Sistema VTON Modular Local",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['identity', 'wardrobe', 'vton'],
        required=True,
        help='Modo de operação'
    )
    
    parser.add_argument(
        '--action',
        help='Ação a executar (depende do modo)'
    )
    
    parser.add_argument(
        '--image',
        type=Path,
        help='Caminho para imagem de entrada'
    )
    
    parser.add_argument(
        '--person',
        type=Path,
        help='Caminho para imagem da pessoa (modo VTON)'
    )
    
    parser.add_argument(
        '--garment',
        help='ID do item de roupa (modo VTON)'
    )
    
    parser.add_argument(
        '--identity-id',
        help='ID da identidade'
    )
    
    parser.add_argument(
        '--item-id',
        help='ID do item de roupa'
    )
    
    parser.add_argument(
        '--category',
        help='Categoria do item (ex: shirt, pants, dress)'
    )
    
    parser.add_argument(
        '--prompt',
        help='Prompt para geração (modo VTON)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Caminho para arquivo de configuração (padrão: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializar sistema
        config, gpu_manager, identity_manager, wardrobe_manager, vton_pipeline = setup_system(args.config)
        
        # Executar modo selecionado
        if args.mode == 'identity':
            mode_identity(args, identity_manager)
        elif args.mode == 'wardrobe':
            mode_wardrobe(args, wardrobe_manager)
        elif args.mode == 'vton':
            mode_vton(args, vton_pipeline)
        
        # Log de uso de memória final
        mem_info = gpu_manager.get_memory_usage()
        if mem_info.get('device') == 'cuda':
            logger.info(f"\n=== Uso de Memória ===")
            logger.info(f"Utilização: {mem_info.get('utilization_pct', 0):.1f}%")
            logger.info(f"Livre: {mem_info.get('free_gb', 0):.2f} GB")
        
    except KeyboardInterrupt:
        logger.info("\n\nOperação cancelada pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
