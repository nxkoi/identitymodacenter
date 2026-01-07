"""
Módulo Core - Utilitários centrais do sistema VTON
Gerenciamento de GPU, configuração e logging
"""

from pathlib import Path
import yaml
import logging
from typing import Dict, Any

class Config:
    """Gerenciador de configuração do sistema"""
    
    def __init__(self, config_path: Path = None):
        """
        Inicializa o gerenciador de configuração
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega a configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
            return {}
    
    def _setup_logging(self):
        """Configura o sistema de logging"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('log_file', 'logs/vton.log')
        
        # Criar diretório de logs se não existir
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém um valor da configuração
        
        Args:
            key: Chave da configuração (suporta notação de ponto)
            default: Valor padrão se a chave não existir
        
        Returns:
            Valor da configuração
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
