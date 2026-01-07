"""
Testes para o módulo Wardrobe
"""

import unittest
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

from src.wardrobe import WardrobeManager, WardrobeItem


class TestWardrobeItem(unittest.TestCase):
    """Testes para WardrobeItem"""
    
    def test_creation(self):
        """Testa criação de item"""
        item = WardrobeItem(
            item_id="test_001",
            image_path=Path("/tmp/test.jpg"),
            category="shirt",
            metadata={'color': 'blue'}
        )
        
        self.assertEqual(item.item_id, "test_001")
        self.assertEqual(item.category, "shirt")
        self.assertEqual(item.metadata['color'], 'blue')
    
    def test_to_dict(self):
        """Testa conversão para dicionário"""
        item = WardrobeItem(
            item_id="test_001",
            image_path=Path("/tmp/test.jpg"),
            category="shirt"
        )
        
        data = item.to_dict()
        self.assertEqual(data['item_id'], "test_001")
        self.assertEqual(data['category'], "shirt")
    
    def test_from_dict(self):
        """Testa criação a partir de dicionário"""
        data = {
            'item_id': 'test_001',
            'image_path': '/tmp/test.jpg',
            'category': 'shirt',
            'metadata': {},
            'created_at': '2024-01-01T00:00:00'
        }
        
        item = WardrobeItem.from_dict(data)
        self.assertEqual(item.item_id, 'test_001')
        self.assertEqual(item.category, 'shirt')


class TestWardrobeManager(unittest.TestCase):
    """Testes para WardrobeManager"""
    
    def setUp(self):
        """Configuração antes de cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "wardrobe"
        self.manager = WardrobeManager(
            storage_path=self.storage_path,
            max_items=10
        )
    
    def _create_test_image(self) -> Image.Image:
        """Cria uma imagem de teste"""
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(array)
    
    def test_initialization(self):
        """Testa inicialização do gerenciador"""
        self.assertTrue(self.storage_path.exists())
        self.assertTrue((self.storage_path / "images").exists())
        self.assertEqual(len(self.manager.items), 0)
    
    def test_add_item(self):
        """Testa adição de item"""
        image = self._create_test_image()
        item_id = self.manager.add_item(
            image=image,
            category="shirt",
            metadata={'color': 'blue'}
        )
        
        self.assertIsNotNone(item_id)
        self.assertIn(item_id, self.manager.items)
        self.assertEqual(len(self.manager.items), 1)
    
    def test_get_item(self):
        """Testa obtenção de item"""
        image = self._create_test_image()
        item_id = self.manager.add_item(image, category="shirt")
        
        item = self.manager.get_item(item_id)
        self.assertIsNotNone(item)
        self.assertEqual(item.item_id, item_id)
    
    def test_get_item_image(self):
        """Testa carregamento de imagem do item"""
        image = self._create_test_image()
        item_id = self.manager.add_item(image, category="shirt")
        
        loaded_image = self.manager.get_item_image(item_id)
        self.assertIsNotNone(loaded_image)
        self.assertIsInstance(loaded_image, Image.Image)
    
    def test_remove_item(self):
        """Testa remoção de item"""
        image = self._create_test_image()
        item_id = self.manager.add_item(image, category="shirt")
        
        result = self.manager.remove_item(item_id)
        self.assertTrue(result)
        self.assertNotIn(item_id, self.manager.items)
        self.assertEqual(len(self.manager.items), 0)
    
    def test_list_items(self):
        """Testa listagem de itens"""
        # Adicionar múltiplos itens
        for i in range(3):
            image = self._create_test_image()
            self.manager.add_item(image, category="shirt")
        
        items = self.manager.list_items()
        self.assertEqual(len(items), 3)
    
    def test_list_items_by_category(self):
        """Testa listagem filtrada por categoria"""
        # Adicionar itens de diferentes categorias
        for _ in range(2):
            image = self._create_test_image()
            self.manager.add_item(image, category="shirt")
        
        for _ in range(3):
            image = self._create_test_image()
            self.manager.add_item(image, category="pants")
        
        shirts = self.manager.list_items(category="shirt")
        pants = self.manager.list_items(category="pants")
        
        self.assertEqual(len(shirts), 2)
        self.assertEqual(len(pants), 3)
    
    def test_max_items_limit(self):
        """Testa limite de itens"""
        # Adicionar itens até o limite
        for i in range(10):
            image = self._create_test_image()
            self.manager.add_item(image, category="shirt")
        
        # Tentar adicionar além do limite
        image = self._create_test_image()
        with self.assertRaises(ValueError):
            self.manager.add_item(image, category="shirt")
    
    def test_get_categories(self):
        """Testa obtenção de categorias"""
        image = self._create_test_image()
        self.manager.add_item(image, category="shirt")
        self.manager.add_item(image, category="pants")
        self.manager.add_item(image, category="shirt")
        
        categories = self.manager.get_categories()
        self.assertEqual(len(categories), 2)
        self.assertIn("shirt", categories)
        self.assertIn("pants", categories)
    
    def test_get_stats(self):
        """Testa obtenção de estatísticas"""
        for _ in range(2):
            image = self._create_test_image()
            self.manager.add_item(image, category="shirt")
        
        for _ in range(3):
            image = self._create_test_image()
            self.manager.add_item(image, category="pants")
        
        stats = self.manager.get_stats()
        
        self.assertEqual(stats['total_items'], 5)
        self.assertEqual(stats['max_items'], 10)
        self.assertEqual(stats['categories']['shirt'], 2)
        self.assertEqual(stats['categories']['pants'], 3)


if __name__ == '__main__':
    unittest.main()
