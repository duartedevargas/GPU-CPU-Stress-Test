import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import click
import psutil
import subprocess

def get_gpu_usage():
    """
    Obtém informações sobre o uso da GPU usando o comando nvidia-smi.

    Returns:
        dict: Um dicionário com informações sobre o uso da GPU.
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            fields = line.split(', ')
            gpu_info.append({
                'name': fields[0],
                'utilization': fields[1],
                'memory_total': fields[2],
                'memory_free': fields[3],
                'memory_used': fields[4]
            })
        return gpu_info
    except subprocess.CalledProcessError:
        print("nvidia-smi não está disponível. Verifique se o driver da GPU está instalado.")
        return []

def get_cpu_usage():
    """
    Obtém informações sobre o uso da CPU.

    Returns:
        dict: Um dicionário com informações sobre o uso da CPU.
    """
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)  # Percentual de uso médio da CPU

    return {
        'cpu_count': cpu_count,
        'cpu_usage': cpu_usage
    }

def check_device_usage(device):
    """
    Verifica o uso do dispositivo especificado (GPU ou CPU).

    Args:
        device (str): Dispositivo a ser monitorado ('cpu' ou 'cuda').

    Returns:
        dict: Informações sobre o uso do dispositivo.
    """
    if device == 'cuda' and torch.cuda.is_available():
        print("Verificando uso da GPU...")
        gpu_usage = get_gpu_usage()
        for i, gpu in enumerate(gpu_usage):
            print(f"GPU {i}:")
            print(f"  Nome: {gpu['name']}")
            print(f"  Utilização: {gpu['utilization']}%")
            print(f"  Memória Total: {gpu['memory_total']} MB")
            print(f"  Memória Livre: {gpu['memory_free']} MB")
            print(f"  Memória Usada: {gpu['memory_used']} MB")
    else:
        print("Verificando uso da CPU...")
        cpu_usage = get_cpu_usage()
        print(f"Número de Núcleos da CPU: {cpu_usage['cpu_count']}")
        print(f"Uso da CPU: {cpu_usage['cpu_usage']}%")

def generate_random_data(num_samples, input_dim, seed=42):
    """
    Gera dados aleatórios para entrada no modelo VAE.

    Args:
        num_samples (int): Número de amostras a serem geradas.
        input_dim (int): Dimensão das entradas.
        seed (int): Semente para garantir reprodutibilidade.

    Returns:
        torch.Tensor: Tensor contendo os dados aleatórios gerados.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Gerar dados aleatórios com distribuição normal padrão
    random_data = np.random.randn(num_samples, input_dim)

    # Padronizar os dados para ter média 0 e desvio padrão 1
    mean = np.mean(random_data, axis=0)
    std = np.std(random_data, axis=0)
    standardized_data = (random_data - mean) / std

    # Normalizar os dados para o intervalo [0, 1]
    normalized_data = (standardized_data - np.min(standardized_data)) / (np.max(standardized_data) - np.min(standardized_data))

    return torch.tensor(normalized_data).float()

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation in PyTorch.
    """

    def __init__(self, x_dim, hidden_dim, z_dim=10):
        super(VAE, self).__init__()

        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_layer2_logvar = nn.Linear(hidden_dim, z_dim)

        # Define decoding layers
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x):
        x = F.relu(self.enc_layer1(x))
        mu = self.enc_layer2_mu(x)
        logvar = self.enc_layer2_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decoder(self, z):
        output = F.relu(self.dec_layer1(z))
        output = torch.sigmoid(self.dec_layer2(output))
        return output

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar

    def loss_function(self, output, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(output, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss



def train_model(X_train, X_val, device, learning_rate=1e-4, batch_size=128, num_epochs=15, hidden_dim=256, latent_dim=50):
    # Garantir que X_train e X_val são tensores e movê-los para o dispositivo
    X_train_tensor = X_train.to(device)
    X_val_tensor = X_val.to(device)

    # Criação do modelo
    x_dim = X_train_tensor.shape[1]
    model = VAE(x_dim=x_dim, hidden_dim=hidden_dim, z_dim=latent_dim).to(device)

    # Se houver mais de uma GPU disponível, usar nn.DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Criar datasets e loaders
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0]
            output, z, mu, logvar = model(x)
            
            # Acessando a função de perda corretamente
            if isinstance(model, nn.DataParallel):
                loss = model.module.loss_function(output, x, mu, logvar)
            else:
                loss = model.loss_function(output, x, mu, logvar)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                output, z, mu, logvar = model(x)
                
                # Acessando a função de perda corretamente
                if isinstance(model, nn.DataParallel):
                    loss = model.module.loss_function(output, x, mu, logvar)
                else:
                    loss = model.loss_function(output, x, mu, logvar)
                
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader.dataset))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader.dataset)}, Val Loss: {val_loss / len(val_loader.dataset)}")

    check_device_usage(device)
    return model, train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

@click.command()
@click.option('--device', default='cuda', help="Device to use: 'cpu' or 'cuda'")
@click.option('--learning_rate', default=1e-4, help="Learning rate for the optimizer")
@click.option('--batch_size', default=128, help="Batch size for training")
@click.option('--num_epochs', default=20, help="Number of epochs for training")
@click.option('--hidden_dim', default=500, help="Dimension of the hidden layer")
@click.option('--latent_dim', default=100, help="Dimension of the latent space")
def main(device, learning_rate, batch_size, num_epochs, hidden_dim, latent_dim):
    data = generate_random_data(num_samples=100000, input_dim=5000, seed=42)
    split_index = int(0.8 * len(data))
    data_train = data[:split_index]
    data_test = data[split_index:]

    X_train_full = data_train
    X_test = data_test

    train_size = int(0.8 * len(X_train_full))
    val_size = len(X_train_full) - train_size
    X_train, X_val = random_split(X_train_full, [train_size, val_size])


    # Converter os dados para tensores de forma correta
    X_train_tensor = torch.stack([data.clone().detach().float() for data in X_train]).to(device)
    X_val_tensor = torch.stack([data.clone().detach().float() for data in X_val]).to(device)
 
    # Inicia a contagem do tempo
    start_time = time.time()

    # Chamar o treinamento do modelo
    model, train_losses, val_losses = train_model(
        X_train=X_train_tensor,
        X_val=X_val_tensor,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    )


    # Fim da contagem do tempo
    end_time = time.time()

    # Calcula o tempo total e imprime
    training_time = end_time - start_time
    print(f"Tempo total de treinamento: {training_time:.2f} segundos")

if __name__ == "__main__":
    main()

