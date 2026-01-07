# Arquitetura Técnica - Sistema VTON Modular

## 📋 Visão Geral

Sistema de Virtual Try-On modular desenvolvido em Python, otimizado para GPUs com VRAM limitada (RTX 4070, 8GB).

## 🏗️ Arquitetura de Componentes

```
┌─────────────────────────────────────────────────────────┐
│                      main.py (CLI)                       │
│                  Interface do Usuário                    │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌─────────────┐ ┌────────────┐
│   Identity     │ │  Wardrobe   │ │    VTON    │
│   Manager      │ │   Manager   │ │  Pipeline  │
└────────┬───────┘ └──────┬──────┘ └─────┬──────┘
         │                │              │
         │         ┌──────┴──────┐       │
         │         │             │       │
         ▼         ▼             ▼       ▼
┌─────────────────────┐   ┌──────────────────┐
│  Temporal Graph     │   │   GPU Manager    │
│  (NetworkX)         │   │  (VRAM Control)  │
└─────────────────────┘   └──────────────────┘
         │                        │
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│           Config Manager                     │
│        (config.yaml + logging)               │
└─────────────────────────────────────────────┘
```

## 🧩 Módulos Principais

### 1. Core (`src/core/`)

#### Config (`config.py`)
- **Responsabilidade**: Gerenciamento de configuração
- **Funcionalidades**:
  - Carregamento de `config.yaml`
  - Acesso a configurações aninhadas
  - Setup de logging
- **Padrão**: Singleton implícito

#### GPU Manager (`gpu_manager.py`)
- **Responsabilidade**: Gestão de recursos GPU
- **Funcionalidades**:
  - Detecção automática de GPU/CPU
  - Float16 automático
  - CPU offload para modelos
  - Captura de OOM errors
  - Carregamento exclusivo de modelos
  - Context managers seguros
- **Otimizações**:
  - `torch.cuda.empty_cache()`
  - `gc.collect()`
  - Offload automático

### 2. Identity (`src/identity/`)

#### Temporal Graph (`temporal_graph.py`)
- **Responsabilidade**: Grafo temporal de identidades
- **Estrutura de Dados**:
  ```python
  {
    'identity_id': [
      (embedding, timestamp),
      (embedding, timestamp),
      ...
    ]
  }
  ```
- **Algoritmos**:
  - **Time Decay**: `weight = decay_factor ^ hours_diff`
  - **Weighted Average**: `avg = Σ(embedding * weight) / Σ(weight)`
  - **Cosine Similarity**: `sim = dot(a,b) / (norm(a) * norm(b))`
- **Características**:
  - Histórico limitado (max_history)
  - Reconhecimento automático
  - Persistência em pickle

#### Identity Manager (`manager.py`)
- **Responsabilidade**: Interface de alto nível
- **Integração**:
  - InsightFace (face detection + embedding)
  - Temporal Graph
  - GPU Manager
- **Fluxo de Trabalho**:
  1. Carregar InsightFace (lazy loading)
  2. Extrair face embedding
  3. Adicionar ao grafo temporal
  4. Salvar estado
  5. Descarregar modelo

### 3. Wardrobe (`src/wardrobe/`)

#### Wardrobe Manager (`manager.py`)
- **Responsabilidade**: Gestão de itens de vestuário
- **Estrutura de Armazenamento**:
  ```
  data/wardrobe/
  ├── images/
  │   ├── item_0001.png
  │   ├── item_0002.png
  │   └── ...
  └── index.json
  ```
- **Operações**:
  - CRUD completo de itens
  - Categorização
  - Metadados customizados
  - Limite de itens
- **Persistência**: JSON + PNG files

### 4. VTON (`src/vton/`)

#### VTON Pipeline (`pipeline.py`)
- **Responsabilidade**: Pipeline de try-on virtual
- **Integração**:
  - Stable Diffusion 1.5 (inpainting)
  - Identity Manager
  - Wardrobe Manager
  - GPU Manager
- **Otimizações SD**:
  - `enable_model_cpu_offload()`
  - `enable_attention_slicing(1)`
  - `enable_vae_slicing()`
  - Float16 variant
- **Fluxo de Processamento**:
  1. Descarregar InsightFace
  2. Carregar SD1.5
  3. Preparar imagens
  4. Gerar máscara
  5. Inpainting
  6. Salvar resultado
  7. Descarregar SD1.5

## 🔄 Fluxo de Dados

### Registro de Identidade

```
Imagem → InsightFace → Face Embedding (512D)
                            │
                            ▼
                    Temporal Graph
                    (média ponderada)
                            │
                            ▼
                    Salvar em disco
                  (data/identity/graph.pkl)
```

### Try-On Virtual

```
Pessoa + Roupa → Preparação → SD1.5 Inpainting → Resultado
     │              │              │                  │
     │              │              ▼                  │
     │              │        GPU Context              │
     │              │        (safe_execution)         │
     │              │                                 │
     │              ▼                                 ▼
     │      Redimensionamento                   outputs/
     │      Geração de máscara              tryon_*.png
     │
     └─> Identity Embedding (opcional)
         para melhor preservação
```

## 🧠 Grafo Temporal - Deep Dive

### Estrutura Matemática

Para uma identidade `i` com observações `{e₁, e₂, ..., eₙ}` nos tempos `{t₁, t₂, ..., tₙ}`:

**Peso temporal**:
```
w(t) = α^(Δt/3600)
onde:
  α = time_decay_factor (0.95 padrão)
  Δt = tempo em segundos desde observação
```

**Embedding médio ponderado**:
```
E_avg = Σ(eᵢ × wᵢ) / Σ(wᵢ)
onde:
  wᵢ = w(t_current - tᵢ)
```

**Decisão de identificação**:
```
if cosine_similarity(e_new, E_avg) > threshold:
    return existing_identity
else:
    return new_identity
```

### Propriedades

1. **Adaptabilidade**: Identidades evoluem com o tempo
2. **Robustez**: Média de múltiplas observações
3. **Esquecimento**: Observações antigas têm menos peso
4. **Eficiência**: Histórico limitado evita crescimento infinito

## ⚡ Otimizações de VRAM

### Estratégias Implementadas

| Estratégia | Economia | Implementação |
|-----------|----------|---------------|
| Float16 | ~50% | `torch_dtype=torch.float16` |
| CPU Offload | ~70% | `enable_model_cpu_offload()` |
| Attention Slicing | ~30% | `enable_attention_slicing(1)` |
| VAE Slicing | ~15% | `enable_vae_slicing()` |
| Carregamento Exclusivo | 100% do 2º modelo | Context managers |
| Limpeza Agressiva | ~10-20% | `empty_cache()` + `gc.collect()` |

### Uso de Memória Típico

**RTX 4070 (8GB VRAM):**
```
Base (SO + drivers):        ~500 MB
InsightFace:               ~1.5 GB
SD1.5 (float16 + offload): ~3.0 GB
Buffers de inferência:     ~1.0 GB
Margem de segurança:       ~2.0 GB
────────────────────────────────────
Total máximo:              ~8.0 GB ✓
```

## 🔒 Garantias de Segurança

### Carregamento Exclusivo

```python
# GPU Manager garante que isso não acontece:
❌ InsightFace (GPU) + SD1.5 (GPU) = OOM

# Sistema força:
✓ InsightFace (GPU) → offload → SD1.5 (GPU)
```

### Captura de OOM

```python
try:
    # Operação GPU
    with gpu_manager.safe_execution():
        result = model(input)
except MemoryError:
    # Recuperação automática:
    # 1. Liberar todos os modelos
    # 2. Limpar cache
    # 3. Notificar usuário
    pass
```

## 📊 Padrões de Design

### 1. Lazy Loading
- Modelos só carregados quando necessário
- Economia de memória inicial
- Startup rápido

### 2. Context Managers
- Gerenciamento automático de recursos
- Cleanup garantido
- Código mais limpo

### 3. Dependency Injection
- Managers recebem dependências
- Facilita testes
- Baixo acoplamento

### 4. Repository Pattern
- Wardrobe Manager abstrai persistência
- Temporal Graph abstrai estrutura de dados
- Fácil substituição de backend

## 🧪 Testing Strategy

### Níveis de Teste

1. **Unit Tests**: Funções individuais
   - test_core.py
   - test_identity.py
   - test_wardrobe.py

2. **Integration Tests**: Interação entre módulos
   - example.py (teste manual)

3. **Validation Tests**: Requisitos do sistema
   - verify_requirements.py

### Cobertura

- ✓ Config loading
- ✓ GPU detection
- ✓ Memory management
- ✓ Temporal graph algorithms
- ✓ CRUD operations
- ✓ Persistence

## 🚀 Performance

### Benchmarks Estimados (RTX 4070)

| Operação | Tempo | VRAM |
|----------|-------|------|
| Face Detection | ~100ms | 1.5GB |
| Embedding Extraction | ~50ms | 1.5GB |
| Identity Recognition | ~10ms | <100MB |
| SD1.5 Inpainting (30 steps) | ~10s | 3.5GB |
| Model Loading (InsightFace) | ~3s | 1.5GB |
| Model Loading (SD1.5) | ~15s | 3.0GB |
| Model Offload | ~2s | -95% |

### Gargalos

1. **Loading de modelos**: Primeira execução baixa da HuggingFace
2. **Inferência SD**: Dominante no tempo total
3. **I/O de disco**: Leitura/escrita de imagens

### Otimizações Futuras

- [ ] Quantização INT8
- [ ] TensorRT compilation
- [ ] Batch processing
- [ ] Model caching
- [ ] Streaming pipeline

## 📝 Convenções de Código

### Python
- PEP 8 compliant
- Type hints onde aplicável
- Docstrings em Português
- Pathlib para todos os caminhos

### Estrutura
```python
"""
Descrição do módulo em Português
"""

import standard_lib
import third_party
from . import local

# Constantes
CONSTANT = value

class ClassName:
    """Docstring em Português"""
    
    def __init__(self, params):
        """
        Inicializa...
        
        Args:
            params: Descrição em Português
        """
        pass
```

## 🔮 Roadmap

### v0.2.0
- [ ] Segmentação automática de corpo
- [ ] Suporte a múltiplos rostos
- [ ] Cache de embeddings

### v0.3.0
- [ ] UI Web (Gradio/Streamlit)
- [ ] API REST
- [ ] Suporte a vídeos

### v1.0.0
- [ ] Produção ready
- [ ] Docker deployment
- [ ] Performance optimizations
- [ ] Comprehensive docs

## 📚 Referências

- **InsightFace**: https://github.com/deepinsight/insightface
- **Stable Diffusion**: https://github.com/Stability-AI/stablediffusion
- **Diffusers**: https://github.com/huggingface/diffusers
- **NetworkX**: https://networkx.org/
- **PyTorch**: https://pytorch.org/
