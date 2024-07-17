import argparse
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrivacyParams:
    def __init__(self, eps: float = 0.5, sensitivity: float = 1.0, mean: float = 0):
        self.eps: float = eps
        self.sensitivity: float = sensitivity
        self.mean: float = mean
        self.var: float = self.sensitivity / self.eps


class Net(nn.Module):
    def __init__(self, privacy_params: PrivacyParams) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.privacy_params: PrivacyParams = privacy_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output: torch.Tensor = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch) -> None:
    model.train()
    # ローカルモデルに読み込み
    ## モデルのインスタンスを作成（同じモデルの定義が必要）
    local_model = Net(model.privacy_params)
    ## グローバルモデルのパラメータを読み込み
    local_model.load_state_dict(model.state_dict())
    local_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output: torch.Tensor = local_model(data)
        loss: torch.Tensor = F.nll_loss(output, target)
        loss.backward()
        # 計算された勾配を保存
        grads: list[torch.Tensor] = [
            param.grad.clone()
            for param in local_model.parameters()
            if param.grad is not None
        ]

        noised_grads: list[torch.Tensor] = []
        privacy_params: PrivacyParams = model.privacy_params

        for grad in grads:
            noise: torch.Tensor = torch.distributions.Laplace(
                privacy_params.mean, privacy_params.var
            ).sample(grad.shape)
            noised_grads.append(grad + noise)

        # オプティマイザのパラメータに手動で勾配を設定
        for param, grad in zip(model.parameters(), noised_grads):
            param.grad = grad.to(device)

        # パラメータの更新
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader) -> int:
    model.eval()
    # ローカルモデルに読み込み
    ## モデルのインスタンスを作成（同じモデルの定義が必要）
    local_model = Net(model.privacy_params)
    ## グローバルモデルのパラメータを読み込み
    local_model.load_state_dict(model.state_dict())
    local_model.eval()
    test_loss: float = 0
    correct: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output: torch.Tensor = local_model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred: torch.Tensor = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += int(pred.eq(target.view_as(pred)).sum().item())

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return correct


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args: argparse.Namespace = parser.parse_args()
    use_cuda: bool = not args.no_cuda and torch.cuda.is_available()
    use_mps: bool = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs: dict[str, Any] = {"batch_size": args.batch_size}
    test_kwargs: dict[str, Any] = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs: dict[str, Any] = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    privacy_params = PrivacyParams(eps=0.5, sensitivity=1.0, mean=0)
    model: Net = Net(privacy_params).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    eps_values: list[float] = [0.1 * i for i in range(1, 3)]
    all_correct_lists: list[list[int]] = []
    noDP_data: list[int] = [
        1122,
        3119,
        4230,
        4861,
        5003,
        5201,
        5350,
        5496,
        5546,
        5623,
        5675,
        5698,
        5704,
        5716,
    ]

    for eps in eps_values:
        privacy_params = PrivacyParams(eps=eps, sensitivity=1.0, mean=0)
        model: Net = Net(privacy_params).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        correct_list: list[int] = []

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            correct: int = test(model, device, test_loader)
            correct_list.append(correct)
            scheduler.step()

        all_correct_lists.append(correct_list)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # グラフの作成
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    axes = axes.flatten()

    for i, correct_list in enumerate(all_correct_lists):
        eps: float = eps_values[i]
        axes[i].plot(
            range(1, args.epochs + 1),
            correct_list,
            marker="o",
            label=f"With DP eps={eps:.1f}",
        )
        axes[i].plot(
            range(1, len(noDP_data) + 1),
            noDP_data,
            marker="x",
            label="No DP (Raw Data)",
        )
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Correct")
        axes[i].set_ylim([0, 10000])
        axes[i].set_title(f"eps={eps:.1f}")
        axes[i].legend()
        axes[i].grid()

    plt.suptitle(
        "Impact of Laplace Mechanism for DP on Model Accuracy"
    )  # 全体のタイトルを設定
    plt.tight_layout(
        rect=(0, 0, 1, 0.97)
    )  # 全体のタイトルを表示するためにレイアウトを調整
    plt.show()


if __name__ == "__main__":
    main()
