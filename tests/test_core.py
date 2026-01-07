"""
Testes para o módulo Core
"""

import unittest
import torch
from pathlib import Path
import tempfile
import yaml

from src.core import Config, GPUManager


class TestConfig(unittest.TestCase):
    """Testes para Config"""
    
    def setUp(self):
        """Configuração antes de cada teste"""
        # Criar arquivo de config temporário
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        test_config = {
            'hardware': {
                'device': 'cuda',
                'use_float16': True
            },
            'identity': {
                'embedding_dim': 512
            },
            'logging': {
                'level': 'INFO',
                'log_file': f'{self.temp_dir}/test.log'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def test_load_config(self):
        """Testa carregamento de configuração"""
        config = Config(self.config_path)
        self.assertIsNotNone(config.config)
        self.assertEqual(config.get('hardware.device'), 'cuda')
    
    def test_get_nested_key(self):
        """Testa acesso a chaves aninhadas"""
        config = Config(self.config_path)
        self.assertEqual(config.get('identity.embedding_dim'), 512)
    
    def test_get_default_value(self):
        """Testa valor padrão para chave inexistente"""
        config = Config(self.config_path)
        self.assertEqual(config.get('nonexistent.key', 'default'), 'default')


class TestGPUManager(unittest.TestCase):
    """Testes para GPUManager"""
    
    def test_initialization(self):
        """Testa inicialização do GPUManager"""
        gpu_manager = GPUManager(use_float16=True, cpu_offload=True)
        self.assertIsNotNone(gpu_manager.device)
        self.assertEqual(gpu_manager.cpu_offload, True)
    
    def test_dtype_selection(self):
        """Testa seleção de dtype"""
        gpu_manager = GPUManager(use_float16=True)
        if gpu_manager.device.type == 'cuda':
            self.assertEqual(gpu_manager.dtype, torch.float16)
        else:
            self.assertEqual(gpu_manager.dtype, torch.float32)
    
    def test_memory_info(self):
        """Testa obtenção de informações de memória"""
        gpu_manager = GPUManager()
        mem_info = gpu_manager.get_memory_usage()
        self.assertIsNotNone(mem_info)
        self.assertIn('device', mem_info)
    
    def test_clear_memory(self):
        """Testa limpeza de memória"""
        gpu_manager = GPUManager()
        # Não deve lançar exceção
        gpu_manager.clear_memory()
    
    def test_exclusive_loading(self):
        """Testa garantia de carregamento exclusivo"""
        gpu_manager = GPUManager()
        
        # Simular modelo carregado
        gpu_manager.current_model = 'insightface'
        
        # Tentar carregar modelo incompatível deve lançar exceção
        with self.assertRaises(RuntimeError):
            gpu_manager.ensure_exclusive_loading('sd15', ['insightface'])


if __name__ == '__main__':
    unittest.main()
