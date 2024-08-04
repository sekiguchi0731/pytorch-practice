import argparse
import math
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
    def __init__(self, eps: float = 0.5, delta: float = 0.00001, mean: float = 0) -> None:
        self.eps: float = eps
        self.delta: float = delta
        self.mean: float = mean
        self.std: float = math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps


class Net(nn.Module):
    def __init__(self, privacy_params: PrivacyParams) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.privacy_params: PrivacyParams = privacy_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output: torch.Tensor = F.log_softmax(x, dim=1)
        return output


def clip_gradients(grads: list[torch.Tensor], clipping_norm: float = 1.0) -> list[torch.Tensor]:
    clipped_grads: list[torch.Tensor] = []
    for grad in grads:
        param_norm: float = grad.norm(2).item() ** 2
        clip_coef: float = clipping_norm / (math.sqrt(param_norm) + 1e-6)
        if clip_coef < 1:
            grad = grad.clone() * clip_coef
        clipped_grads.append(grad)
    return clipped_grads


def add_gaussian_noise(model, grads: list[torch.Tensor]) -> list[torch.Tensor]:
    privacy_params: PrivacyParams = model.privacy_params
    noised_grads: list[torch.Tensor] = []
    for grad in grads:
        noise: torch.Tensor = torch.randn(grad.shape) * privacy_params.std + privacy_params.mean
        noised_grads.append(grad + noise.to(grad.device))
    return noised_grads


def train(args, model, device, train_loader, optimizer, epoch) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # バッチ全体の出力を取得
        output: torch.Tensor = model(data)
        loss: torch.Tensor = F.nll_loss(output, target, reduction="none")

        # 勾配のリストを初期化
        grads_per_data: list[list[torch.Tensor]] = []

        # 各データポイントごとに勾配を計算
        for i in range(len(data)):
            model.zero_grad()
            loss[i].backward(retain_graph=True)
            grads: list[torch.Tensor] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            grads_per_data.append(grads)

        # 各データポイントの勾配をクリッピング
        # 64x8x([32,1,3,3],[32],...)
        clipped_grads_per_data: list[list[torch.Tensor]] = [
            clip_gradients(grads) for grads in grads_per_data
        ]

        # クリッピングされた勾配の平均を計算
        avg_grads: list[torch.Tensor] = [
            torch.mean(
                torch.stack([grads[i] for grads in clipped_grads_per_data]), dim=0
            )
            for i in range(len(clipped_grads_per_data[0]))
        ]

        # noised_grads: list[torch.Tensor] = avg_grads
        # ノイズの追加
        noised_grads: list[torch.Tensor] = add_gaussian_noise(model, avg_grads)

        # 平均化された勾配をモデルに適用
        for param, noised_grad in zip(model.parameters(), noised_grads):
            if param.grad is not None:
                param.grad.data.copy_(noised_grad)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.mean().item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader) -> int:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

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

def plot_graph(
    args,
    eps_values: list[float],
    all_correct_lists_without_scaling: list[list[int]],
    all_correct_lists_with_scaling: list[list[int]] = [],
) -> None:
    # グラフの作成
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    axes = axes.flatten()

    for i, correct_list in enumerate(all_correct_lists_without_scaling):
        eps: float = eps_values[i]
        noDP_list: list[int] = [
            9821,
            9871,
            9887,
            9900,
            9899,
            9906,
            9907,
            9912,
            9906,
            9919,
            9916,
            9914,
            9918,
            9917,
        ]
        axes[i].plot(
            range(1, args.epochs + 1),
            correct_list,
            marker="x",
            label="With DP",
        )
        axes[i].plot(
            range(1, args.epochs + 1),
            noDP_list,
            marker="o",
            label="Without DP",
        )
        # axes[i].plot(
        #     range(1, args.epochs + 1),
        #     all_correct_lists_with_scaling[i],
        #     marker="o",
        #     label="With Scaling",
        # )

        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Correct")
        axes[i].set_ylim([0, 10000])
        axes[i].set_title(f"eps={eps:.2f}")
        axes[i].legend()
        axes[i].grid()

    plt.suptitle(r"Impact of DP on Model Accuracy")
    # plt.suptitle(
    #     r"Impact of Laplace Mechanism for DP on Model Accuracy with $\sqrt{m}$ Scaling"
    # )

    plt.tight_layout(
        rect=(0, 0, 1, 0.97)
    )  # 全体のタイトルを表示するためにレイアウトを調整
    plt.show()


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

    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    train_kwargs: dict[str, Any] = {"batch_size": args.batch_size}
    test_kwargs: dict[str, Any] = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    eps_values: list[float] = [0.5 * i for i in range(1, 2)]
    all_collect_lists: list[list[int]] = []
    for eps in eps_values:
        privacy_params = PrivacyParams(eps=eps, delta=0.00001, mean=0)
        model: Net = Net(privacy_params).to(device)
        print(eps, privacy_params.delta, privacy_params.std)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        collect_list: list[int] = []

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            collect_list.append(test(model, device, test_loader))
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

        all_collect_lists.append(collect_list)

    plot_graph(args, eps_values, all_collect_lists)


if __name__ == "__main__":
    main()
