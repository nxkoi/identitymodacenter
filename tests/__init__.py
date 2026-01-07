"""
Testes iniciais - verifica estrutura básica
"""

import unittest
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports(unittest.TestCase):
    """Testa se todos os módulos podem ser importados"""
    
    def test_import_core(self):
        """Testa importação do módulo core"""
        try:
            from src.core import Config, GPUManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Falha ao importar core: {e}")
    
    def test_import_identity(self):
        """Testa importação do módulo identity"""
        try:
            from src.identity import TemporalIdentityGraph, IdentityManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Falha ao importar identity: {e}")
    
    def test_import_wardrobe(self):
        """Testa importação do módulo wardrobe"""
        try:
            from src.wardrobe import WardrobeManager, WardrobeItem
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Falha ao importar wardrobe: {e}")
    
    def test_import_vton(self):
        """Testa importação do módulo vton"""
        try:
            from src.vton import VTONPipeline
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Falha ao importar vton: {e}")


if __name__ == '__main__':
    unittest.main()
