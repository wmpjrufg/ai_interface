# Redes Neurais Profundas para Dados Tabulares

## 1. Da para usar rede neural em dataset tabular?

Sim. E possivel usar redes neurais profundas em datasets tabulares, tanto para regressao quanto para classificacao.

Porem, para dados tabulares, modelos baseados em arvores costumam ser muito fortes, especialmente em datasets pequenos ou medios. Bons modelos de comparacao sao:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

Uma rede neural profunda pode ser interessante quando:

- o dataset tem muitas amostras;
- existem muitas variaveis numericas e categoricas;
- ha interacoes complexas entre atributos;
- ha variaveis categoricas com alta cardinalidade;
- o modelo tabular precisa ser combinado com imagens, texto, series temporais ou outros tipos de dados.

## 2. O que muda entre regressao e classificacao?

A arquitetura interna da rede pode ser muito parecida. O que muda principalmente e:

- a camada final;
- a funcao de perda;
- o formato do alvo;
- a forma de interpretar a saida;
- as metricas de avaliacao.

Em outras palavras, o "corpo" da rede aprende uma representacao dos dados. A ultima camada transforma essa representacao no tipo de resposta desejada.

## 3. Comparacao entre tipos de problema

| Tipo de problema | Saida final | Loss comum | Ativacao na inferencia |
|---|---:|---|---|
| Regressao | 1 ou mais valores continuos | `MSELoss`, `L1Loss`, `HuberLoss` | geralmente nenhuma |
| Classificacao binaria | 1 logit | `BCEWithLogitsLoss` | `sigmoid` |
| Classificacao multiclasse | `n_classes` logits | `CrossEntropyLoss` | `softmax` |
| Classificacao multilabel | `n_labels` logits | `BCEWithLogitsLoss` | `sigmoid` por classe |

## 4. Exemplo de rede neural tabular profunda generica

Abaixo esta um exemplo de uma rede neural profunda para dados tabulares que pode ser usada em regressao, classificacao binaria ou classificacao multiclasse.

```python
import torch
import torch.nn as nn


class DeepTabularNN(nn.Module):
    def __init__(
        self,
        n_features,
        task_type,
        n_classes=None,
        n_targets=1,
        hidden_layers=(512, 256, 128, 64),
        dropout=0.25
    ):
        super().__init__()

        self.task_type = task_type

        layers = []
        in_dim = n_features

        for h in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        if task_type == "regression":
            out_dim = n_targets
        elif task_type == "binary":
            out_dim = 1
        elif task_type == "multiclass":
            out_dim = n_classes
        else:
            raise ValueError(
                "task_type deve ser 'regression', 'binary' ou 'multiclass'"
            )

        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

## 5. Uso para regressao

Em regressao, a saida geralmente tem um neuronio, ou mais de um caso existam varios alvos.

```python
model = DeepTabularNN(
    n_features=20,
    task_type="regression",
    n_targets=1
)

loss_fn = nn.MSELoss()
```

Metricas comuns:

- MAE;
- RMSE;
- R2.

Em regressao, e comum normalizar tambem a variavel alvo `y`, principalmente quando seus valores possuem escala muito grande.

## 6. Uso para classificacao binaria

Em classificacao binaria, a camada final deve ter apenas uma saida.

```python
model = DeepTabularNN(
    n_features=20,
    task_type="binary"
)

loss_fn = nn.BCEWithLogitsLoss()
```

A saida do modelo e um logit. Para transformar em probabilidade:

```python
logits = model(X)
probs = torch.sigmoid(logits)
preds = (probs >= 0.5).int()
```

Metricas comuns:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- matriz de confusao.

## 7. Uso para classificacao multiclasse

Quando existem mais de duas classes e cada amostra pertence a apenas uma classe, o problema e de classificacao multiclasse.

Exemplo: classes `baixo`, `medio`, `alto` e `critico`.

Nesse caso, a rede deve ter uma saida por classe.

```python
n_classes = 4

model = DeepTabularNN(
    n_features=20,
    task_type="multiclass",
    n_classes=n_classes
)

loss_fn = nn.CrossEntropyLoss()
```

O alvo `y` deve ser um inteiro representando a classe:

```python
y = torch.tensor([0, 3, 1, 2], dtype=torch.long)
```

Durante o treino, nao se deve aplicar `softmax` antes da `CrossEntropyLoss`, pois ela ja espera logits brutos.

Para obter as probabilidades na inferencia:

```python
logits = model(X)
probs = torch.softmax(logits, dim=1)
preds = torch.argmax(probs, dim=1)
```

Metricas recomendadas:

- accuracy;
- macro F1;
- balanced accuracy;
- precision por classe;
- recall por classe;
- matriz de confusao.

## 8. E se as classes forem desbalanceadas?

Se uma classe aparece muito mais que as outras, a rede pode aprender a prever sempre a classe majoritaria.

Nesse caso, pode-se usar pesos na funcao de perda.

```python
weights = torch.tensor([1.0, 2.5, 4.0, 3.0])
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

Tambem e importante avaliar com metricas que considerem o desbalanceamento, como:

- macro F1;
- balanced accuracy;
- matriz de confusao;
- recall por classe.

## 9. Configuracao para uma plataforma

Para uma plataforma onde usuarios escolhem o tipo de problema, uma configuracao simples poderia ser:

```python
config = {
    "task_type": "multiclass",
    "target_column": "classe",
    "hidden_layers": [512, 256, 128, 64],
    "dropout": 0.25,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
}
```

A plataforma pode usar `task_type` para decidir automaticamente:

- quantos neuronios terao na camada final;
- qual loss sera usada;
- como converter a saida em predicao;
- quais metricas calcular.

## 10. Tamanhos sugeridos de rede

Uma ideia simples e oferecer opcoes predefinidas:

```python
architectures = {
    "pequena": [64, 32],
    "media": [128, 64, 32],
    "profunda": [512, 256, 128, 64],
    "muito_profunda": [1024, 512, 256, 128, 64]
}
```

Para uma opcao padrao segura:

```python
hidden_layers = [256, 128, 64]
dropout = 0.2
weight_decay = 1e-4
early_stopping = True
```

## 11. Cuidados importantes

Redes neurais profundas podem sofrer overfitting em datasets tabulares pequenos.

Por isso, recomenda-se usar:

- normalizacao das variaveis numericas;
- codificacao adequada das variaveis categoricas;
- dropout;
- weight decay;
- early stopping;
- validacao separada;
- comparacao com modelos baseline, como CatBoost ou LightGBM.

## 12. Resumo final

Uma rede neural profunda para dados tabulares pode ser bastante generica. A parte central da rede pode ser a mesma para regressao e classificacao.

O que muda e principalmente a ultima camada, a loss e a interpretacao da saida:

- regressao: prever valores continuos;
- binaria: prever uma probabilidade com `sigmoid`;
- multiclasse: prever uma classe entre varias com `softmax`;
- multilabel: prever varias classes independentes com `sigmoid`.

Para uma plataforma, o ideal e esconder essa complexidade do usuario e deixar que ele escolha apenas:

- tipo de problema;
- coluna alvo;
- tamanho da rede;
- parametros basicos de treino.

Por baixo, a plataforma configura automaticamente a rede, a loss, as metricas e o pos-processamento.
