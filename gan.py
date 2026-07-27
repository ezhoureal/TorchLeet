import torch
import torch.nn as nn

B = 8
D = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D * 4), nn.ReLU(), nn.Linear(D * 4, D))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Linear(D, 2)

    def forward(self, x):
        return self.dis(x)


def gan():
    generator = Generator()
    discriminator = Discriminator()

    real_labels = torch.arange(0, 2, dtype=torch.float).unsqueeze(0).expand(B, 2)
    fake_labels = torch.arange(1, -1, -1, dtype=torch.float).unsqueeze(0).expand(B, 2)
    print(f"fake labels = {fake_labels}")

    optimizer_g = torch.optim.AdamW(generator.parameters())
    optimizer_d = torch.optim.AdamW(discriminator.parameters())

    criterion = nn.CrossEntropyLoss()

    for i in range(100):
        real_data = torch.cos(torch.rand(B, D)).log_softmax(-1)
        fake_data = generator(torch.rand(B, D)).detach()
        discriminator_loss = criterion(
            discriminator(real_data), real_labels
        ) + criterion(discriminator(fake_data), fake_labels)
        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()

        fake_data = generator(torch.rand(B, D))
        logits_fake = discriminator(fake_data)
        loss = criterion(logits_fake, real_labels)

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        print(f"G loss = {loss}, D loss = {discriminator_loss}")


if __name__ == "__main__":
    gan()
