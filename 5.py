import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pandas as pd # type: ignore

# Импорт пользовательского оптимизатора FTML
class FTML(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1=0.6, beta2=0.999, epsilon=1e-8):
        # Проверка на валидность входных параметров
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon parameter: {}".format(epsilon))
        # Инициализация параметров оптимизатора
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(FTML, self).__init__(params, defaults)

def step(self, closure=None):
    loss = None
    if closure is not None:
        loss = closure()

    # Проход по всем группам параметров
    for group in self.param_groups:
        # Проход по всем параметрам в группе
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]

            # Инициализация состояния, если оно пустое
            if len(state) == 0:
                state['t'] = 0
                state['v_hat'] = torch.zeros_like(p.data)
                state['z'] = torch.zeros_like(p.data)
                state['d'] = torch.zeros_like(p.data)

            state['t'] += 1
            t = state['t']
            beta1, beta2 = group['beta1'], group['beta2']
            epsilon = group['epsilon']

            # Обновление состояний v_hat и z
            state['v_hat'] = beta2 * state['v_hat'] + (1 - beta2) * grad
            state['z'] = beta1 * state['z'] + (1 - beta1) * grad

            # Коррекция смещения для v_hat и z
            v_hat_bias_corr = state['v_hat'] / (1 - beta2 ** t)
            z_bias_corr = state['z'] / (1 - beta1 ** t)

            # Вычисление направления d и обновление состояния d
            d = -state['z'] / (torch.sqrt(v_hat_bias_corr) + epsilon)
            state['d'] = beta1 * state['d'] + (1 - beta1) * d

            # Обновление параметра p
            p.data += group['lr'] * state['d']

    return loss

# Загрузка данных и преобразование
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_data = MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(mnist_data))
test_size = len(mnist_data) - train_size
train_data, test_data = random_split(mnist_data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Определение модели
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        # Определение слоев модели
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Прямой проход модели
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Обучение модели с алгоритмом FTML
def train_model(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Оценка модели на тестовой выборке
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy}")
    return accuracy

# Параметры модели и обучение
input_size = 28 * 28
output_size = 10
epochs = 10
learning_rate = 0.001

# for hidden_size1 in [5, 10, 15, 20, 25]:
#     for hidden_size2 in [5, 10, 15, 20, 25]:
#         model = MLP(input_size, hidden_size1, hidden_size2, output_size)
#         optimizer = FTML(model.parameters(), lr=learning_rate)  # Использование пользовательского оптимизатора FTML
#         criterion = nn.CrossEntropyLoss()
#         print(f"\nTraining model with hidden layer sizes: {hidden_size1}, {hidden_size2}")
#         train_model(model, train_loader, optimizer, criterion, epochs=epochs)
#         test_model(model, test_loader)

# Фрагмент кода для заполнения данных точности моделей в таблицу
hidden_sizes = [5, 10, 15, 20, 25]
data = {hs2: [] for hs2 in hidden_sizes}
index = []

for hs1 in hidden_sizes:
    index.append(hs1)
    for hs2 in hidden_sizes:
        model = MLP(input_size, hs1, hs2, output_size)
        optimizer = FTML(model.parameters(), lr=learning_rate)  # Использование пользовательского оптимизатора FTML
        criterion = nn.CrossEntropyLoss()
        print(f"\nTraining model with hidden layer sizes: {hs1}, {hs2}")
        train_model(model, train_loader, optimizer, criterion, epochs=epochs)
        accuracy = test_model(model, test_loader)
        data[hs2].append(accuracy)

df = pd.DataFrame(data, index=index, columns=hidden_sizes)

# Вывод таблицы данных
print(df)