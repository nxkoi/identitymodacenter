# Identity Moda Center - Sistema VTON Modular

Sistema de Virtual Try-On (VTON) modular local otimizado para RTX 4070 (8GB VRAM).

## 🎯 Características

- **Arquitetura Modular**: Módulos independentes (identity, wardrobe, vton)
- **Otimizado para VRAM Limitada**: Float16, CPU offload, carregamento exclusivo de modelos
- **Gestão Inteligente de Identidade**: Grafo temporal com média de embeddings e time decay
- **Proteção contra OOM**: Captura e gerenciamento de erros de memória GPU
- **Segurança de Carregamento**: InsightFace e SD1.5 nunca carregados juntos na GPU

## 🏗️ Arquitetura

```
src/
├── core/              # Utilitários centrais
│   ├── config.py      # Gerenciamento de configuração
│   └── gpu_manager.py # Gerenciamento de GPU e memória
├── identity/          # Gestão de identidade
│   ├── temporal_graph.py  # Grafo temporal
│   └── manager.py         # Gerenciador de identidade
├── wardrobe/          # Guarda-roupa virtual
│   └── manager.py     # Gerenciador de itens
└── vton/              # Pipeline VTON
    └── pipeline.py    # Pipeline principal
```

## 📋 Requisitos

- Python 3.8+
- CUDA 11.8+ (para GPU)
- RTX 4070 ou similar (8GB+ VRAM recomendado)

## 🚀 Instalação

```bash
# Clonar repositório
git clone https://github.com/nxkoi/identitymodacenter.git
cd identitymodacenter

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 💻 Uso

### Modo Identidade

```bash
# Registrar nova identidade
python main.py --mode identity --action register --image pessoa.jpg

# Identificar pessoa em imagem
python main.py --mode identity --action identify --image nova_foto.jpg

# Listar identidades registradas
python main.py --mode identity --action list
```

### Modo Guarda-roupa

```bash
# Adicionar item ao guarda-roupa
python main.py --mode wardrobe --action add --image camisa.jpg --category shirt

# Listar itens
python main.py --mode wardrobe --action list

# Ver estatísticas
python main.py --mode wardrobe --action stats

# Remover item
python main.py --mode wardrobe --action remove --item-id item_0001
```

### Modo VTON (Virtual Try-On)

```bash
# Try-on básico
python main.py --mode vton --person pessoa.jpg --garment item_0001

# Try-on com identidade registrada
python main.py --mode vton --person pessoa.jpg --garment item_0001 --identity-id identity_0000

# Try-on com prompt customizado
python main.py --mode vton --person pessoa.jpg --garment item_0001 --prompt "pessoa vestindo camisa elegante"
```

## ⚙️ Configuração

Edite `config.yaml` para ajustar:

- Limites de VRAM
- Parâmetros do grafo temporal
- Configurações de modelos
- Caminhos de armazenamento

## 🧠 Grafo Temporal de Identidade

O sistema usa um grafo temporal para gerenciar identidades:

- **Time Decay**: Embeddings mais antigos têm peso menor
- **Média Ponderada**: Identidade representada pela média ponderada de observações
- **Reconhecimento Automático**: Identifica pessoas automaticamente por similaridade
- **Histórico Limitado**: Mantém apenas N observações mais recentes

## 🔧 Otimizações de VRAM

### Estratégias Implementadas

1. **Float16**: Reduz uso de memória pela metade
2. **CPU Offload**: Move modelos para CPU quando não em uso
3. **Carregamento Exclusivo**: Apenas um modelo pesado na GPU por vez
4. **Attention Slicing**: Reduz uso de memória no Stable Diffusion
5. **VAE Slicing**: Otimização adicional para VAE
6. **Captura de OOM**: Recuperação automática de erros de memória

### Gerenciamento de Modelos

```python
# InsightFace e SD1.5 nunca são carregados juntos
# O sistema garante isso automaticamente
```

## 📊 Estrutura de Dados

```
data/
├── identity/
│   └── graph.pkl          # Grafo temporal de identidades
├── wardrobe/
│   ├── images/            # Imagens de itens
│   └── index.json         # Índice de itens
└── outputs/               # Resultados de try-on
```

## 🐛 Troubleshooting

### Erro de OOM (Out of Memory)

Se encontrar erros de memória:

1. Reduza `image_size` em `config.yaml`
2. Reduza `num_inference_steps`
3. Ative `cpu_offload` (já ativo por padrão)
4. Use `float16` (já ativo por padrão)

### Modelos não carregam

Verifique:

1. CUDA está instalado corretamente
2. Drivers NVIDIA atualizados
3. Espaço em disco para download de modelos

## 📝 Licença

MIT

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 👥 Autores

- Sistema desenvolvido para otimização em RTX 4070
- Arquitetura modular e grafo temporal