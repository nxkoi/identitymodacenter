# Guia de Uso - Sistema VTON Modular

## 📚 Índice

1. [Instalação](#instalação)
2. [Configuração](#configuração)
3. [Uso Básico](#uso-básico)
4. [Casos de Uso Avançados](#casos-de-uso-avançados)
5. [Otimizações](#otimizações)
6. [Solução de Problemas](#solução-de-problemas)

## 🔧 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- CUDA 11.8+ (para uso com GPU)
- 8GB+ VRAM (RTX 4070 ou similar)
- 20GB+ espaço em disco (para modelos)

### Passo a Passo

```bash
# 1. Clonar o repositório
git clone https://github.com/nxkoi/identitymodacenter.git
cd identitymodacenter

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. (Opcional) Instalar em modo desenvolvimento
pip install -e .
```

### Verificação da Instalação

```bash
# Executar script de exemplo
python example.py

# Verificar requisitos
python verify_requirements.py
```

## ⚙️ Configuração

### Arquivo config.yaml

Edite `config.yaml` para personalizar o sistema:

```yaml
# Configurações de Hardware
hardware:
  device: "cuda"          # "cuda" ou "cpu"
  vram_limit_gb: 8
  use_float16: true       # Economiza 50% de VRAM
  cpu_offload: true       # Move modelos para CPU quando não em uso

# Configurações de Identidade
identity:
  temporal_graph:
    time_decay_factor: 0.95      # Quanto menor, maior o decaimento
    max_history: 100             # Máximo de observações por identidade
    similarity_threshold: 0.7    # Limiar para reconhecimento (0-1)

# Configurações de VTON
vton:
  image_size: [512, 768]         # [largura, altura]
  num_inference_steps: 30        # Mais steps = melhor qualidade
  guidance_scale: 7.5            # Controle de aderência ao prompt
```

### Otimizações de VRAM

Para GPUs com menos VRAM:

```yaml
# Configuração mínima (6GB VRAM)
vton:
  image_size: [384, 576]         # Reduzir resolução
  num_inference_steps: 20        # Menos steps
```

## 🎯 Uso Básico

### 1. Gerenciamento de Identidade

#### Registrar Nova Identidade

```bash
python main.py --mode identity --action register --image foto_pessoa.jpg
```

Saída esperada:
```
✓ Identidade registrada: identity_0000
```

#### Identificar Pessoa em Foto

```bash
python main.py --mode identity --action identify --image nova_foto.jpg
```

Saída esperada:
```
✓ Pessoa identificada: identity_0000
# ou
✗ Nenhuma identidade reconhecida
```

#### Listar Identidades

```bash
python main.py --mode identity --action list
```

Saída esperada:
```
=== Identidades Registradas (3) ===
  - identity_0000: 5 observações
  - identity_0001: 2 observações
  - identity_0002: 1 observações
```

### 2. Gerenciamento de Guarda-roupa

#### Adicionar Item

```bash
# Adicionar camisa
python main.py --mode wardrobe --action add \
  --image camisa_azul.jpg \
  --category shirt

# Adicionar calça
python main.py --mode wardrobe --action add \
  --image calca_jeans.jpg \
  --category pants
```

#### Listar Itens

```bash
# Listar todos
python main.py --mode wardrobe --action list

# Listar por categoria
python main.py --mode wardrobe --action list --category shirt
```

#### Ver Estatísticas

```bash
python main.py --mode wardrobe --action stats
```

Saída esperada:
```
=== Estatísticas do Guarda-roupa ===
  Total de itens: 15/1000
  Categorias:
    - shirt: 8
    - pants: 5
    - dress: 2
```

#### Remover Item

```bash
python main.py --mode wardrobe --action remove --item-id item_0001_20240101120000
```

### 3. Virtual Try-On

#### Try-on Básico

```bash
python main.py --mode vton \
  --person pessoa.jpg \
  --garment item_0001_20240101120000
```

#### Try-on com Identidade Registrada

```bash
python main.py --mode vton \
  --person pessoa.jpg \
  --garment item_0001_20240101120000 \
  --identity-id identity_0000
```

#### Try-on com Prompt Customizado

```bash
python main.py --mode vton \
  --person pessoa.jpg \
  --garment item_0001_20240101120000 \
  --prompt "pessoa elegante vestindo camisa social azul, foto profissional, alta qualidade"
```

## 🚀 Casos de Uso Avançados

### Uso Programático

```python
from pathlib import Path
from PIL import Image
from src.core import Config, GPUManager
from src.identity import IdentityManager
from src.wardrobe import WardrobeManager
from src.vton import VTONPipeline

# Inicializar sistema
config = Config()
gpu_manager = GPUManager(use_float16=True, cpu_offload=True)

identity_manager = IdentityManager(
    gpu_manager=gpu_manager,
    config=config.config
)

wardrobe_manager = WardrobeManager(
    storage_path=Path("data/wardrobe"),
    max_items=1000
)

vton_pipeline = VTONPipeline(
    gpu_manager=gpu_manager,
    identity_manager=identity_manager,
    wardrobe_manager=wardrobe_manager,
    config=config.config
)

# Registrar identidade
person_image = Image.open("pessoa.jpg")
identity_id = identity_manager.register_identity(person_image)

# Adicionar roupa
garment_image = Image.open("camisa.jpg")
garment_id = wardrobe_manager.add_item(garment_image, category="shirt")

# Realizar try-on
result = vton_pipeline.try_on_with_identity(
    identity_id=identity_id,
    garment_item_id=garment_id,
    base_image=person_image
)

if result:
    result.save("resultado.png")
```

### Processamento em Lote

```python
from pathlib import Path
from PIL import Image

# Processar múltiplas imagens
image_dir = Path("fotos")
for image_path in image_dir.glob("*.jpg"):
    image = Image.open(image_path)
    identity_id = identity_manager.identify_person(image)
    print(f"{image_path.name}: {identity_id}")
```

### Gerenciamento de Memória Manual

```python
# Liberar memória explicitamente
gpu_manager.clear_memory()

# Descarregar modelos específicos
identity_manager.unload_model()
vton_pipeline.unload_model()

# Verificar uso de memória
mem_info = gpu_manager.get_memory_usage()
print(f"VRAM livre: {mem_info['free_gb']:.2f} GB")
```

## 🔍 Otimizações

### Para RTX 4070 (8GB VRAM)

Configuração recomendada já está em `config.yaml`:
- ✓ Float16 ativo
- ✓ CPU Offload ativo
- ✓ Carregamento exclusivo
- ✓ Attention/VAE slicing

### Para GPUs com Menos VRAM (6GB)

```yaml
vton:
  image_size: [384, 576]
  num_inference_steps: 20
  
hardware:
  cpu_offload: true
  use_float16: true
```

### Para GPUs com Mais VRAM (12GB+)

```yaml
vton:
  image_size: [768, 1024]
  num_inference_steps: 50
  
hardware:
  cpu_offload: false  # Mantém modelos na GPU
```

## 🐛 Solução de Problemas

### Erro: Out of Memory (OOM)

**Sintoma**: `CUDA out of memory` ou `GPU OOM`

**Soluções**:
1. Reduzir resolução em `config.yaml`
2. Reduzir `num_inference_steps`
3. Garantir que `cpu_offload: true`
4. Fechar outros programas usando GPU

```bash
# Verificar memória GPU
nvidia-smi
```

### Erro: Modelo não carrega

**Sintoma**: `Failed to load model`

**Soluções**:
1. Verificar conexão com internet (primeira vez baixa modelos)
2. Verificar espaço em disco (~20GB necessário)
3. Verificar instalação do CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Erro: Face não detectada

**Sintoma**: `Nenhuma face detectada na imagem`

**Soluções**:
1. Verificar qualidade da imagem
2. Garantir que face está visível e frontal
3. Verificar iluminação adequada
4. Testar com outra imagem

### Performance lenta

**Soluções**:
1. Usar GPU ao invés de CPU
2. Verificar se CUDA está instalado corretamente
3. Reduzir `num_inference_steps`
4. Usar resolução menor

### Identidades não reconhecidas

**Sintoma**: Sempre cria novas identidades

**Ajustar threshold em `config.yaml`**:
```yaml
identity:
  temporal_graph:
    similarity_threshold: 0.6  # Valor menor = mais permissivo
```

## 📊 Monitoramento

### Logs

Logs são salvos em `logs/vton.log`:

```bash
tail -f logs/vton.log
```

### Métricas de Memória

```python
# No código
mem_info = gpu_manager.get_memory_usage()
print(f"Utilização: {mem_info['utilization_pct']:.1f}%")
print(f"Livre: {mem_info['free_gb']:.2f} GB")
```

## 🔒 Boas Práticas

1. **Sempre use CPU offload** em GPUs com 8GB ou menos
2. **Salve o grafo temporal regularmente** (automático após cada operação)
3. **Monitore uso de memória** durante operações longas
4. **Use resolução apropriada** para sua GPU
5. **Faça backup** de `data/` periodicamente

## 📝 Exemplos Completos

### Pipeline Completo

```bash
# 1. Registrar pessoa
python main.py --mode identity --action register --image pessoa1.jpg

# 2. Adicionar roupas
python main.py --mode wardrobe --action add --image camisa1.jpg --category shirt
python main.py --mode wardrobe --action add --image calca1.jpg --category pants

# 3. Listar itens
python main.py --mode wardrobe --action list

# 4. Try-on
python main.py --mode vton \
  --person pessoa1.jpg \
  --garment item_0000_... \
  --identity-id identity_0000

# 5. Verificar resultado em outputs/
ls outputs/
```

## 🆘 Suporte

Para problemas não listados aqui:
1. Verifique os logs em `logs/vton.log`
2. Execute `python verify_requirements.py`
3. Abra uma issue no GitHub
4. Inclua informações do sistema e logs relevantes
