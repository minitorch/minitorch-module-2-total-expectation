"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # x shape: (batch_size, 2)
        # Layer1: Linear(2, hidden_layers) -> ReLU
        x = self.layer1(x)
        x = x.relu()

        # Layer2: Linear(hidden_layers, hidden_layers) -> ReLU
        x = self.layer2(x)
        x = x.relu()

        # Layer3: Linear(hidden_layers, 1) -> Sigmoid
        x = self.layer3(x)
        x = minitorch.tensor_functions.Sigmoid.apply(x)

        return x


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        weights_tensor = self.weights.value
        bias_tensor = self.bias.value

        batch_size = x.shape[0]
        out_size = weights_tensor.shape[1]

        # Initialize output tensor
        result = minitorch.zeros((batch_size, out_size))

        # Manual matrix multiplication: y = x @ W^T + b
        for i in range(batch_size):  # For each sample in batch
            for j in range(out_size):  # For each output neuron
                # Compute dot product: x[i] @ weights_tensor[:, j]
                dot_product = 0
                for k in range(x.shape[1]):  # For each input feature
                    dot_product = dot_product + x[i, k] * weights_tensor[k, j]
                result[i, j] = dot_product + bias_tensor[j]

        return result


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
