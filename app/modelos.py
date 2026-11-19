"""Copias EXACTAS de las clases del notebook (celdas 21 y 29). Si cambian allá, deben cambiar aquí."""
import torch.nn as nn

N_CLASES = 3


class PerceptronMulticapa(nn.Module):
    """Capas ocultas lineales con ReLU y dropout. La salida son 3 logits (softmax en la pérdida)."""
    def __init__(self, n_entradas, capas, dropout):
        super().__init__()
        bloques, anterior = [], n_entradas
        for neuronas in capas:
            bloques += [nn.Linear(anterior, neuronas), nn.ReLU(), nn.Dropout(dropout)]
            anterior = neuronas
        bloques.append(nn.Linear(anterior, N_CLASES))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


class RedRecurrente(nn.Module):
    """LSTM o GRU sobre la ventana de semanas; clasifica con el estado oculto de la última semana."""
    def __init__(self, n_entradas, celda, unidades, capas, dropout):
        super().__init__()
        Celda = nn.LSTM if celda == "LSTM" else nn.GRU
        self.rnn = Celda(n_entradas, unidades, num_layers=capas, batch_first=True,
                         dropout=dropout if capas > 1 else 0.0)
        self.salida = nn.Sequential(nn.Dropout(dropout), nn.Linear(unidades, N_CLASES))

    def forward(self, x):
        salida, _ = self.rnn(x)
        return self.salida(salida[:, -1, :])
