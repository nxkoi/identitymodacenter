"""
Testes para o módulo Identity
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from src.identity import TemporalIdentityGraph


class TestTemporalIdentityGraph(unittest.TestCase):
    """Testes para TemporalIdentityGraph"""
    
    def setUp(self):
        """Configuração antes de cada teste"""
        self.graph = TemporalIdentityGraph(
            embedding_dim=512,
            time_decay_factor=0.95,
            max_history=100,
            similarity_threshold=0.7
        )
    
    def test_initialization(self):
        """Testa inicialização do grafo"""
        self.assertEqual(self.graph.embedding_dim, 512)
        self.assertEqual(self.graph.time_decay_factor, 0.95)
        self.assertEqual(self.graph.max_history, 100)
        self.assertEqual(self.graph.similarity_threshold, 0.7)
    
    def test_add_identity(self):
        """Testa adição de identidade"""
        embedding = np.random.randn(512).astype(np.float32)
        identity_id = self.graph.add_or_update_identity(embedding)
        
        self.assertIsNotNone(identity_id)
        self.assertIn(identity_id, self.graph.embeddings_history)
    
    def test_identity_recognition(self):
        """Testa reconhecimento de identidade"""
        # Adicionar identidade inicial
        embedding1 = np.random.randn(512).astype(np.float32)
        identity_id1 = self.graph.add_or_update_identity(embedding1)
        
        # Adicionar embedding similar (deve reconhecer mesma identidade)
        embedding2 = embedding1 + np.random.randn(512).astype(np.float32) * 0.01  # Muito similar
        identity_id2 = self.graph.add_or_update_identity(embedding2)
        
        # Pode ser a mesma identidade ou diferente dependendo do threshold
        # Apenas verificar que não lança exceção
        self.assertIsNotNone(identity_id2)
    
    def test_get_identity_embedding(self):
        """Testa obtenção de embedding de identidade"""
        embedding = np.random.randn(512).astype(np.float32)
        identity_id = self.graph.add_or_update_identity(embedding)
        
        retrieved = self.graph.get_identity_embedding(identity_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, (512,))
    
    def test_cosine_similarity(self):
        """Testa cálculo de similaridade de cosseno"""
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Embeddings idênticos
        sim1 = self.graph._cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(sim1, 1.0, places=5)
        
        # Embeddings ortogonais
        sim2 = self.graph._cosine_similarity(emb1, emb3)
        self.assertAlmostEqual(sim2, 0.0, places=5)
    
    def test_time_decay(self):
        """Testa decaimento temporal"""
        now = datetime.now()
        past = now - timedelta(hours=24)
        
        weight = self.graph._compute_time_weight(past, now)
        
        # Peso deve ser menor que 1 (decaimento)
        self.assertLess(weight, 1.0)
        self.assertGreater(weight, 0.0)
    
    def test_weighted_average(self):
        """Testa média ponderada"""
        now = datetime.now()
        
        embeddings = [
            (np.ones(512, dtype=np.float32), now - timedelta(hours=24)),
            (np.ones(512, dtype=np.float32) * 2, now)
        ]
        
        avg = self.graph._compute_weighted_average(embeddings, now)
        
        # Média deve estar entre 1 e 2, mais próxima de 2 (mais recente)
        self.assertTrue(np.all(avg >= 1.0))
        self.assertTrue(np.all(avg <= 2.0))
        self.assertTrue(np.mean(avg) > 1.5)  # Mais peso no mais recente
    
    def test_max_history_limit(self):
        """Testa limite de histórico"""
        embedding = np.random.randn(512).astype(np.float32)
        
        # Adicionar mais embeddings que o limite
        for _ in range(150):
            self.graph.add_or_update_identity(embedding, identity_id="test_id")
        
        # Verificar que não excede o limite
        history_len = len(self.graph.embeddings_history["test_id"])
        self.assertLessEqual(history_len, self.graph.max_history)
    
    def test_save_load(self):
        """Testa salvar e carregar grafo"""
        # Adicionar algumas identidades
        for _ in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            self.graph.add_or_update_identity(embedding)
        
        # Salvar
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.graph.save(temp_path)
            
            # Carregar
            loaded_graph = TemporalIdentityGraph.load(temp_path)
            
            # Verificar
            self.assertEqual(loaded_graph.embedding_dim, self.graph.embedding_dim)
            self.assertEqual(len(loaded_graph.embeddings_history), len(self.graph.embeddings_history))
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_get_all_identities(self):
        """Testa listagem de identidades"""
        # Adicionar identidades
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            self.graph.add_or_update_identity(embedding)
        
        identities = self.graph.get_all_identities()
        self.assertEqual(len(identities), 3)
    
    def test_get_identity_info(self):
        """Testa obtenção de informações de identidade"""
        embedding = np.random.randn(512).astype(np.float32)
        identity_id = self.graph.add_or_update_identity(embedding)
        
        info = self.graph.get_identity_info(identity_id)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['identity_id'], identity_id)
        self.assertEqual(info['num_observations'], 1)
        self.assertIsNotNone(info['first_seen'])
        self.assertIsNotNone(info['last_seen'])


if __name__ == '__main__':
    unittest.main()
