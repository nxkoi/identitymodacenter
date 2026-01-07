"""
Gerenciador de Identidade
Integra InsightFace com Grafo Temporal
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import cv2

from .temporal_graph import TemporalIdentityGraph
from ..core import GPUManager

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Gerenciador de identidade usando InsightFace e Grafo Temporal
    Otimizado para baixa VRAM
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        config: Dict[str, Any],
        graph_save_path: Optional[Path] = None
    ):
        """
        Inicializa o gerenciador de identidade
        
        Args:
            gpu_manager: Gerenciador de GPU
            config: Configuração do sistema
            graph_save_path: Caminho para salvar/carregar o grafo
        """
        self.gpu_manager = gpu_manager
        self.config = config
        self.graph_save_path = graph_save_path or Path("data/identity/graph.pkl")
        
        # InsightFace será carregado sob demanda
        self.face_analysis = None
        
        # Configurações de identidade
        identity_config = config.get('identity', {})
        temporal_config = identity_config.get('temporal_graph', {})
        
        # Inicializar grafo temporal
        if self.graph_save_path.exists():
            logger.info(f"Carregando grafo existente de {self.graph_save_path}")
            self.temporal_graph = TemporalIdentityGraph.load(self.graph_save_path)
        else:
            logger.info("Criando novo grafo temporal")
            self.temporal_graph = TemporalIdentityGraph(
                embedding_dim=identity_config.get('embedding_dim', 512),
                time_decay_factor=temporal_config.get('time_decay_factor', 0.95),
                max_history=temporal_config.get('max_history', 100),
                similarity_threshold=temporal_config.get('similarity_threshold', 0.7)
            )
        
        logger.info("IdentityManager inicializado")
    
    def _load_insightface(self):
        """
        Carrega o modelo InsightFace (apenas quando necessário)
        IMPORTANTE: Nunca carregar junto com SD1.5
        """
        if self.face_analysis is not None:
            return
        
        try:
            # Garantir que SD1.5 não está carregado
            self.gpu_manager.ensure_exclusive_loading('insightface', ['sd15'])
            
            logger.info("Carregando InsightFace...")
            
            # Importar aqui para evitar importação desnecessária
            from insightface.app import FaceAnalysis
            
            # Configuração do InsightFace
            model_config = self.config.get('models', {}).get('insightface', {})
            model_name = model_config.get('name', 'buffalo_l')
            providers = model_config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
            self.face_analysis = FaceAnalysis(
                name=model_name,
                providers=providers
            )
            self.face_analysis.prepare(ctx_id=0 if self.gpu_manager.device.type == 'cuda' else -1)
            
            logger.info("InsightFace carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar InsightFace: {e}")
            raise
    
    def extract_face_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Extrai embedding de face de uma imagem
        
        Args:
            image: Imagem PIL
        
        Returns:
            Embedding de face ou None se nenhuma face for detectada
        """
        try:
            # Carregar InsightFace se necessário
            if self.face_analysis is None:
                self._load_insightface()
            
            # Converter PIL para numpy array (formato BGR para OpenCV)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with self.gpu_manager.safe_execution():
                # Detectar faces
                faces = self.face_analysis.get(img_array)
                
                if len(faces) == 0:
                    logger.warning("Nenhuma face detectada na imagem")
                    return None
                
                # Usar a primeira face detectada
                face = faces[0]
                embedding = face.embedding
                
                logger.info(f"Face embedding extraído: shape={embedding.shape}")
                return embedding
                
        except Exception as e:
            logger.error(f"Erro ao extrair embedding: {e}")
            return None
    
    def register_identity(
        self,
        image: Image.Image,
        identity_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Registra ou atualiza uma identidade
        
        Args:
            image: Imagem da pessoa
            identity_id: ID da identidade (None para detectar automaticamente)
        
        Returns:
            ID da identidade registrada ou None em caso de erro
        """
        embedding = self.extract_face_embedding(image)
        
        if embedding is None:
            return None
        
        # Adicionar ao grafo temporal
        identity_id = self.temporal_graph.add_or_update_identity(
            embedding=embedding,
            identity_id=identity_id
        )
        
        # Salvar grafo
        self.save_graph()
        
        return identity_id
    
    def identify_person(self, image: Image.Image) -> Optional[str]:
        """
        Identifica uma pessoa em uma imagem
        
        Args:
            image: Imagem da pessoa
        
        Returns:
            ID da identidade ou None se não identificado
        """
        embedding = self.extract_face_embedding(image)
        
        if embedding is None:
            return None
        
        # Tentar identificar (None como identity_id faz busca automática)
        identity_id = self.temporal_graph.add_or_update_identity(
            embedding=embedding,
            identity_id=None
        )
        
        # Salvar grafo
        self.save_graph()
        
        return identity_id
    
    def get_identity_embedding(self, identity_id: str) -> Optional[np.ndarray]:
        """
        Obtém o embedding médio de uma identidade
        
        Args:
            identity_id: ID da identidade
        
        Returns:
            Embedding ou None
        """
        return self.temporal_graph.get_identity_embedding(identity_id)
    
    def list_identities(self) -> list:
        """
        Lista todas as identidades registradas
        
        Returns:
            Lista de informações de identidades
        """
        identities = []
        for identity_id in self.temporal_graph.get_all_identities():
            info = self.temporal_graph.get_identity_info(identity_id)
            if info:
                identities.append(info)
        return identities
    
    def save_graph(self):
        """Salva o grafo temporal"""
        self.temporal_graph.save(self.graph_save_path)
    
    def unload_model(self):
        """Descarrega o modelo InsightFace da memória"""
        if self.face_analysis is not None:
            logger.info("Descarregando InsightFace")
            self.face_analysis = None
            self.gpu_manager.clear_memory()
