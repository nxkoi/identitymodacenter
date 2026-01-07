"""
Gerenciador de Guarda-roupa
Armazena e gerencia itens de vestuário
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import json
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class WardrobeItem:
    """Representa um item de vestuário"""
    
    def __init__(
        self,
        item_id: str,
        image_path: Path,
        category: str = "clothing",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa um item de vestuário
        
        Args:
            item_id: ID único do item
            image_path: Caminho para a imagem do item
            category: Categoria do item (ex: camisa, calça, vestido)
            metadata: Metadados adicionais
        """
        self.item_id = item_id
        self.image_path = Path(image_path)
        self.category = category
        self.metadata = metadata or {}
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'item_id': self.item_id,
            'image_path': str(self.image_path),
            'category': self.category,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WardrobeItem':
        """Cria instância a partir de dicionário"""
        item = cls(
            item_id=data['item_id'],
            image_path=Path(data['image_path']),
            category=data.get('category', 'clothing'),
            metadata=data.get('metadata', {})
        )
        if 'created_at' in data:
            item.created_at = datetime.fromisoformat(data['created_at'])
        return item


class WardrobeManager:
    """Gerenciador de guarda-roupa"""
    
    def __init__(self, storage_path: Path, max_items: int = 1000):
        """
        Inicializa o gerenciador de guarda-roupa
        
        Args:
            storage_path: Diretório para armazenar itens
            max_items: Número máximo de itens
        """
        self.storage_path = Path(storage_path)
        self.max_items = max_items
        
        # Criar diretórios
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.images_path = self.storage_path / "images"
        self.images_path.mkdir(exist_ok=True)
        
        # Arquivo de índice
        self.index_path = self.storage_path / "index.json"
        
        # Carregar items
        self.items: Dict[str, WardrobeItem] = {}
        self._load_index()
        
        logger.info(f"WardrobeManager inicializado - {len(self.items)} itens carregados")
    
    def _load_index(self):
        """Carrega o índice de itens"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item_data in data.get('items', []):
                    item = WardrobeItem.from_dict(item_data)
                    self.items[item.item_id] = item
                
                logger.info(f"Índice carregado: {len(self.items)} itens")
            except Exception as e:
                logger.error(f"Erro ao carregar índice: {e}")
                self.items = {}
    
    def _save_index(self):
        """Salva o índice de itens"""
        try:
            data = {
                'items': [item.to_dict() for item in self.items.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Índice salvo")
        except Exception as e:
            logger.error(f"Erro ao salvar índice: {e}")
    
    def add_item(
        self,
        image: Image.Image,
        category: str = "clothing",
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None
    ) -> str:
        """
        Adiciona um item ao guarda-roupa
        
        Args:
            image: Imagem do item
            category: Categoria do item
            metadata: Metadados adicionais
            item_id: ID do item (None para gerar automaticamente)
        
        Returns:
            ID do item adicionado
        
        Raises:
            ValueError: Se exceder o limite de itens
        """
        if len(self.items) >= self.max_items:
            raise ValueError(f"Limite de {self.max_items} itens atingido")
        
        # Gerar ID se não fornecido
        if item_id is None:
            item_id = f"item_{len(self.items):04d}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Salvar imagem
        image_filename = f"{item_id}.png"
        image_path = self.images_path / image_filename
        image.save(image_path, format='PNG')
        
        # Criar item
        item = WardrobeItem(
            item_id=item_id,
            image_path=image_path,
            category=category,
            metadata=metadata
        )
        
        self.items[item_id] = item
        self._save_index()
        
        logger.info(f"Item adicionado: {item_id} (categoria: {category})")
        return item_id
    
    def get_item(self, item_id: str) -> Optional[WardrobeItem]:
        """
        Obtém um item pelo ID
        
        Args:
            item_id: ID do item
        
        Returns:
            WardrobeItem ou None
        """
        return self.items.get(item_id)
    
    def get_item_image(self, item_id: str) -> Optional[Image.Image]:
        """
        Carrega a imagem de um item
        
        Args:
            item_id: ID do item
        
        Returns:
            Imagem PIL ou None
        """
        item = self.get_item(item_id)
        if item is None:
            return None
        
        try:
            return Image.open(item.image_path)
        except Exception as e:
            logger.error(f"Erro ao carregar imagem do item {item_id}: {e}")
            return None
    
    def remove_item(self, item_id: str) -> bool:
        """
        Remove um item do guarda-roupa
        
        Args:
            item_id: ID do item
        
        Returns:
            True se removido, False se não encontrado
        """
        item = self.items.get(item_id)
        if item is None:
            return False
        
        # Remover imagem
        try:
            if item.image_path.exists():
                item.image_path.unlink()
        except Exception as e:
            logger.warning(f"Erro ao remover imagem: {e}")
        
        # Remover do índice
        del self.items[item_id]
        self._save_index()
        
        logger.info(f"Item removido: {item_id}")
        return True
    
    def list_items(
        self,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[WardrobeItem]:
        """
        Lista itens do guarda-roupa
        
        Args:
            category: Filtrar por categoria (None para todos)
            limit: Limitar número de resultados
        
        Returns:
            Lista de WardrobeItem
        """
        items = list(self.items.values())
        
        # Filtrar por categoria
        if category is not None:
            items = [item for item in items if item.category == category]
        
        # Ordenar por data de criação (mais recente primeiro)
        items.sort(key=lambda x: x.created_at, reverse=True)
        
        # Limitar resultados
        if limit is not None:
            items = items[:limit]
        
        return items
    
    def get_categories(self) -> List[str]:
        """
        Retorna lista de categorias disponíveis
        
        Returns:
            Lista de categorias
        """
        categories = set(item.category for item in self.items.values())
        return sorted(categories)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do guarda-roupa
        
        Returns:
            Dicionário com estatísticas
        """
        categories_count = {}
        for item in self.items.values():
            categories_count[item.category] = categories_count.get(item.category, 0) + 1
        
        return {
            'total_items': len(self.items),
            'max_items': self.max_items,
            'categories': categories_count,
            'storage_path': str(self.storage_path)
        }
