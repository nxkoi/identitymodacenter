"""
Gerenciador de GPU - Otimizado para RTX 4070 (8GB VRAM)
Implementa estratégias de economia de memória e proteção contra OOM
"""

import torch
import logging
import gc
from typing import Optional, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUManager:
    """Gerenciador de recursos da GPU com otimizações para baixa VRAM"""
    
    def __init__(self, use_float16: bool = True, cpu_offload: bool = True):
        """
        Inicializa o gerenciador de GPU
        
        Args:
            use_float16: Usar float16 para economizar memória
            cpu_offload: Fazer offload de modelos para CPU quando não estiverem em uso
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_float16 = use_float16 and self.device.type == "cuda"
        self.cpu_offload = cpu_offload
        self.dtype = torch.float16 if self.use_float16 else torch.float32
        
        # Estado dos modelos carregados
        self.loaded_models = {}
        self.current_model = None
        
        logger.info(f"GPUManager inicializado - Device: {self.device}, Float16: {self.use_float16}, CPU Offload: {self.cpu_offload}")
        
        if self.device.type == "cuda":
            self._log_gpu_info()
    
    def _log_gpu_info(self):
        """Registra informações sobre a GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, Memória Total: {total_memory:.2f} GB")
    
    def get_memory_usage(self) -> dict:
        """
        Retorna informações sobre uso de memória da GPU
        
        Returns:
            Dicionário com informações de memória
        """
        if self.device.type != "cuda":
            return {"device": "cpu"}
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "device": "cuda",
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
            "utilization_pct": (reserved / total) * 100
        }
    
    def clear_memory(self):
        """Limpa a memória da GPU"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memória da GPU liberada")
            
            # Log do estado após limpeza
            mem_info = self.get_memory_usage()
            logger.info(f"Memória livre após limpeza: {mem_info['free_gb']:.2f} GB")
    
    @contextmanager
    def model_context(self, model_name: str, model: Any):
        """
        Context manager para carregamento exclusivo de modelos
        Garante que apenas um modelo esteja na GPU por vez
        
        Args:
            model_name: Nome do modelo (ex: 'insightface', 'sd15')
            model: Instância do modelo
        
        Yields:
            Modelo configurado para o device correto
        """
        try:
            # Se há outro modelo carregado, fazer offload para CPU
            if self.current_model and self.current_model != model_name:
                logger.info(f"Fazendo offload do modelo {self.current_model} para CPU")
                self._offload_current_model()
            
            # Carregar modelo na GPU
            logger.info(f"Carregando modelo {model_name} na GPU")
            if hasattr(model, 'to'):
                model = model.to(self.device)
                if self.use_float16 and hasattr(model, 'half'):
                    model = model.half()
            
            self.current_model = model_name
            self.loaded_models[model_name] = model
            
            # Log de memória
            mem_info = self.get_memory_usage()
            logger.info(f"Memória utilizada: {mem_info.get('utilization_pct', 0):.1f}%")
            
            yield model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM Error ao carregar {model_name}")
                self._handle_oom_error()
                raise MemoryError(f"GPU OOM ao carregar {model_name}. Tente reduzir o batch size ou resolução.")
            raise
        
        finally:
            # Se cpu_offload estiver ativo, mover modelo para CPU
            if self.cpu_offload:
                logger.info(f"Fazendo offload do modelo {model_name} para CPU")
                self._offload_model(model_name, model)
    
    def _offload_current_model(self):
        """Move o modelo atual para CPU"""
        if self.current_model and self.current_model in self.loaded_models:
            model = self.loaded_models[self.current_model]
            self._offload_model(self.current_model, model)
    
    def _offload_model(self, model_name: str, model: Any):
        """
        Move um modelo para CPU
        
        Args:
            model_name: Nome do modelo
            model: Instância do modelo
        """
        try:
            if hasattr(model, 'to'):
                model = model.to('cpu')
            self.clear_memory()
        except Exception as e:
            logger.warning(f"Erro ao fazer offload do modelo {model_name}: {e}")
    
    def _handle_oom_error(self):
        """Gerencia erro de OOM (Out of Memory)"""
        logger.warning("Tentando recuperar de OOM error...")
        
        # Liberar todos os modelos
        for model_name in list(self.loaded_models.keys()):
            model = self.loaded_models[model_name]
            self._offload_model(model_name, model)
        
        self.loaded_models.clear()
        self.current_model = None
        self.clear_memory()
    
    @contextmanager
    def safe_execution(self):
        """
        Context manager para execução segura com captura de OOM
        
        Yields:
            None
        
        Raises:
            MemoryError: Se ocorrer OOM
        """
        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM Error durante execução")
                self._handle_oom_error()
                raise MemoryError("GPU Out of Memory. Operação cancelada.")
            raise
    
    def ensure_exclusive_loading(self, current_model_type: str, forbidden_types: list):
        """
        Garante que modelos incompatíveis não sejam carregados juntos
        Por exemplo: InsightFace e SD1.5 nunca devem estar na GPU ao mesmo tempo
        
        Args:
            current_model_type: Tipo do modelo atual (ex: 'insightface', 'sd15')
            forbidden_types: Lista de tipos que não podem coexistir
        
        Raises:
            RuntimeError: Se tentar carregar modelos incompatíveis
        """
        if self.current_model in forbidden_types:
            raise RuntimeError(
                f"Não é possível carregar {current_model_type} enquanto "
                f"{self.current_model} está na GPU. Modelos incompatíveis."
            )
        
        # Fazer offload de qualquer modelo dos tipos proibidos
        for forbidden_type in forbidden_types:
            if forbidden_type in self.loaded_models:
                logger.info(f"Removendo modelo incompatível {forbidden_type}")
                self._offload_model(forbidden_type, self.loaded_models[forbidden_type])
                del self.loaded_models[forbidden_type]
