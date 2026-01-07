"""
Grafo Temporal de Identidade
Gerencia embeddings de identidade com decaimento temporal
"""

import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class TemporalIdentityGraph:
    """
    Grafo temporal para gerenciamento de identidades
    Implementa média de embeddings com time decay
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        time_decay_factor: float = 0.95,
        max_history: int = 100,
        similarity_threshold: float = 0.7
    ):
        """
        Inicializa o grafo temporal de identidade
        
        Args:
            embedding_dim: Dimensão dos embeddings de identidade
            time_decay_factor: Fator de decaimento temporal (0-1)
            max_history: Máximo de embeddings históricos por identidade
            similarity_threshold: Limiar de similaridade para identificação
        """
        self.embedding_dim = embedding_dim
        self.time_decay_factor = time_decay_factor
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        
        # Grafo de identidades
        self.graph = nx.DiGraph()
        
        # Armazenamento de embeddings com timestamps
        # {identity_id: [(embedding, timestamp), ...]}
        self.embeddings_history: Dict[str, List[Tuple[np.ndarray, datetime]]] = {}
        
        # Contador de identidades
        self.next_identity_id = 0
        
        logger.info(
            f"TemporalIdentityGraph inicializado - "
            f"dim={embedding_dim}, decay={time_decay_factor}, "
            f"max_history={max_history}, threshold={similarity_threshold}"
        )
    
    def _compute_time_weight(self, timestamp: datetime, current_time: datetime) -> float:
        """
        Calcula o peso temporal baseado no decaimento
        
        Args:
            timestamp: Timestamp do embedding
            current_time: Timestamp atual
        
        Returns:
            Peso temporal (0-1)
        """
        time_diff = (current_time - timestamp).total_seconds()
        # Decaimento exponencial baseado em horas
        hours_diff = time_diff / 3600.0
        weight = self.time_decay_factor ** hours_diff
        return weight
    
    def _compute_weighted_average(
        self,
        embeddings_with_time: List[Tuple[np.ndarray, datetime]],
        current_time: datetime
    ) -> np.ndarray:
        """
        Calcula a média ponderada de embeddings com decaimento temporal
        
        Args:
            embeddings_with_time: Lista de (embedding, timestamp)
            current_time: Timestamp atual para cálculo de pesos
        
        Returns:
            Embedding médio ponderado
        """
        if not embeddings_with_time:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0.0
        
        for embedding, timestamp in embeddings_with_time:
            weight = self._compute_time_weight(timestamp, current_time)
            weighted_sum += embedding * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return embeddings_with_time[-1][0]  # Retorna o mais recente
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula similaridade de cosseno entre dois embeddings
        
        Args:
            emb1: Primeiro embedding
            emb2: Segundo embedding
        
        Returns:
            Similaridade (0-1)
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def add_or_update_identity(
        self,
        embedding: np.ndarray,
        identity_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Adiciona ou atualiza uma identidade no grafo
        
        Args:
            embedding: Embedding de identidade (face embedding)
            identity_id: ID da identidade (None para criar nova)
            timestamp: Timestamp do embedding (None para usar tempo atual)
        
        Returns:
            ID da identidade
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Normalizar embedding
        embedding = embedding.astype(np.float32)
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Se identity_id não foi fornecido, tentar identificar ou criar novo
        if identity_id is None:
            identity_id = self._identify_or_create(embedding, timestamp)
        
        # Adicionar ao histórico
        if identity_id not in self.embeddings_history:
            self.embeddings_history[identity_id] = []
        
        self.embeddings_history[identity_id].append((embedding, timestamp))
        
        # Limitar histórico
        if len(self.embeddings_history[identity_id]) > self.max_history:
            self.embeddings_history[identity_id] = self.embeddings_history[identity_id][-self.max_history:]
        
        # Atualizar grafo
        if not self.graph.has_node(identity_id):
            self.graph.add_node(identity_id, created_at=timestamp)
        
        self.graph.nodes[identity_id]['last_seen'] = timestamp
        self.graph.nodes[identity_id]['num_observations'] = len(self.embeddings_history[identity_id])
        
        logger.debug(f"Identidade {identity_id} atualizada com {len(self.embeddings_history[identity_id])} observações")
        
        return identity_id
    
    def _identify_or_create(self, embedding: np.ndarray, timestamp: datetime) -> str:
        """
        Identifica uma identidade existente ou cria uma nova
        
        Args:
            embedding: Embedding para identificar
            timestamp: Timestamp atual
        
        Returns:
            ID da identidade
        """
        best_match_id = None
        best_similarity = 0.0
        
        # Comparar com todas as identidades existentes
        for identity_id, history in self.embeddings_history.items():
            # Calcular embedding médio ponderado
            avg_embedding = self._compute_weighted_average(history, timestamp)
            
            # Calcular similaridade
            similarity = self._cosine_similarity(embedding, avg_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = identity_id
        
        # Se similaridade acima do threshold, retornar identidade existente
        if best_similarity >= self.similarity_threshold:
            logger.info(f"Identidade reconhecida: {best_match_id} (similaridade: {best_similarity:.3f})")
            return best_match_id
        
        # Caso contrário, criar nova identidade
        new_id = f"identity_{self.next_identity_id:04d}"
        self.next_identity_id += 1
        logger.info(f"Nova identidade criada: {new_id}")
        return new_id
    
    def get_identity_embedding(
        self,
        identity_id: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[np.ndarray]:
        """
        Obtém o embedding médio ponderado de uma identidade
        
        Args:
            identity_id: ID da identidade
            timestamp: Timestamp para cálculo de pesos (None para usar tempo atual)
        
        Returns:
            Embedding médio ou None se identidade não existir
        """
        if identity_id not in self.embeddings_history:
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        history = self.embeddings_history[identity_id]
        return self._compute_weighted_average(history, timestamp)
    
    def get_all_identities(self) -> List[str]:
        """
        Retorna lista de todas as identidades
        
        Returns:
            Lista de IDs de identidade
        """
        return list(self.embeddings_history.keys())
    
    def get_identity_info(self, identity_id: str) -> Optional[Dict]:
        """
        Obtém informações sobre uma identidade
        
        Args:
            identity_id: ID da identidade
        
        Returns:
            Dicionário com informações ou None
        """
        if identity_id not in self.embeddings_history:
            return None
        
        history = self.embeddings_history[identity_id]
        node_data = self.graph.nodes.get(identity_id, {})
        
        return {
            'identity_id': identity_id,
            'num_observations': len(history),
            'first_seen': history[0][1] if history else None,
            'last_seen': history[-1][1] if history else None,
            'created_at': node_data.get('created_at'),
        }
    
    def save(self, filepath: Path):
        """
        Salva o grafo em disco
        
        Args:
            filepath: Caminho para salvar
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embedding_dim': self.embedding_dim,
            'time_decay_factor': self.time_decay_factor,
            'max_history': self.max_history,
            'similarity_threshold': self.similarity_threshold,
            'next_identity_id': self.next_identity_id,
            'embeddings_history': self.embeddings_history,
            'graph': self.graph
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Grafo salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TemporalIdentityGraph':
        """
        Carrega o grafo do disco
        
        Args:
            filepath: Caminho do arquivo
        
        Returns:
            Instância de TemporalIdentityGraph
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        graph = cls(
            embedding_dim=data['embedding_dim'],
            time_decay_factor=data['time_decay_factor'],
            max_history=data['max_history'],
            similarity_threshold=data['similarity_threshold']
        )
        
        graph.next_identity_id = data['next_identity_id']
        graph.embeddings_history = data['embeddings_history']
        graph.graph = data['graph']
        
        logger.info(f"Grafo carregado de {filepath}")
        return graph
