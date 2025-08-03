import numpy as np
import random
import pickle

class SimpleRNN:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.input_size, self.hidden_size, self.output_size = sizes[0], sizes[1], sizes[2]

        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01
        
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))

    def feedforward(self, x):
        h = np.zeros((self.hidden_size, 1))
        for x_i in x:
            x_i = x_i.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x_i) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_Wxh = np.zeros_like(self.Wxh)
        nabla_Whh = np.zeros_like(self.Whh)
        nabla_Why = np.zeros_like(self.Why)
        nabla_bh = np.zeros_like(self.bh)
        nabla_by = np.zeros_like(self.by)

        for x, y in mini_batch:
            delta_nabla_Wxh, delta_nabla_Whh, delta_nabla_Why, delta_nabla_bh, delta_nabla_by = self.backprop(x, y)
            nabla_Wxh += delta_nabla_Wxh
            nabla_Whh += delta_nabla_Whh
            nabla_Why += delta_nabla_Why
            nabla_bh += delta_nabla_bh
            nabla_by += delta_nabla_by
        
        learning_factor = eta / len(mini_batch)
        self.Wxh -= learning_factor * nabla_Wxh
        self.Whh -= learning_factor * nabla_Whh
        self.Why -= learning_factor * nabla_Why
        self.bh -= learning_factor * nabla_bh
        self.by -= learning_factor * nabla_by
        
    def backprop(self, x, y):
        # Forward pass untuk mendapatkan state
        h = np.zeros((self.hidden_size, 1))
        hs = { -1: h }
        for i, x_i in enumerate(x):
            x_i = x_i.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x_i) + np.dot(self.Whh, hs[i-1]) + self.bh)
            hs[i] = h
        
        logits = np.dot(self.Why, h) + self.by
        
        # Backward pass
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum(axis=0)
        d_logits = probs
        d_logits[y] -= 1

        d_Why = np.dot(d_logits, h.T)
        d_by = d_logits
        
        d_Wxh = np.zeros_like(self.Wxh)
        d_Whh = np.zeros_like(self.Whh)
        d_bh = np.zeros_like(self.bh)
        
        dh_next = np.dot(self.Why.T, d_logits)

        for t in reversed(range(len(x))):
            h_t = hs[t]
            h_prev = hs[t-1]
            dh_raw = (1 - h_t**2) * dh_next
            
            d_bh += dh_raw
            d_Wxh += np.dot(dh_raw, x[t].reshape(1, -1))
            d_Whh += np.dot(dh_raw, h_prev.T)
            
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        for dparam in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(dparam, -5, 5, out=dparam)

        return (d_Wxh, d_Whh, d_Why, d_bh, d_by)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        """Menyimpan model ke dalam file."""
        data = {
            "sizes": self.sizes,
            "weights": [self.Wxh, self.Whh, self.Why],
            "biases": [self.bh, self.by]
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename):
        """Memuat model dari file."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        net = SimpleRNN(data["sizes"])
        net.Wxh, net.Whh, net.Why = data["weights"]
        net.bh, net.by = data["biases"]
        return net