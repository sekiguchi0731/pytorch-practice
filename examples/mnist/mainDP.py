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
    def __init__(self, eps: float = 0.5, sensitivity: float = 1.0, mean: float = 0) -> None:
        self.eps: float = eps
        self.sensitivity: float = sensitivity
        self.mean: float = mean
        self.var: float = self.sensitivity / self.eps


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


# 勾配のクリッピング
## modelとclipping_normを受け取り、model内の勾配を直接変更
def clip_gradients(model, clipping_norm=1.0) -> None:
    # 全てのパラメータについて勾配をクリッピング
    for p in model.parameters():
        if p.grad is not None:
            # print(p.grad[0])
            param_norm: torch.Tensor = p.grad.data.norm(2)
            clip_coef: torch.Tensor = clipping_norm / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


# 勾配の平均化
def average_gradients(grads_per_data: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    num_data_points: int = len(grads_per_data)
    num_params: int = len(grads_per_data[0])
    avg_grads: list[torch.Tensor] = []

    for i in range(num_params):
        for j in range(num_data_points):
            stacked_grads: torch.Tensor = torch.stack([grads_per_data[j][i]])
        avg_grad: torch.Tensor = torch.mean(stacked_grads, dim=0)
        avg_grads.append(avg_grad)

    return avg_grads


def train(args, model, device, train_loader, optimizer, epoch, is_Scaling) -> None:
    model.train()
    for batch_idx, (dataset, targets) in enumerate(train_loader):
        dataset, targets = dataset.to(device), targets.to(device)
        output: torch.Tensor = model(dataset)
        grads_per_data: list[list[torch.Tensor]] = []
        for idx, data in enumerate(dataset):
            optimizer.zero_grad()
            loss: torch.Tensor = F.nll_loss(output[idx], targets[idx])
            # 計算グラフを保持：dataが複数ある場合に、複数回backwardを計算できないから。
            loss.backward(retain_graph=True)
            # print(loss)
            clip_gradients(model)
            grads: list[torch.Tensor] = [
                p.grad.clone() for p in model.parameters() if p.grad is not None
            ]
            grads_per_data.append(grads)

        # print(len(grads_per_data[0][0]))
        # 勾配の平均化
        # 勾配の平均化
        avg_grads: list[torch.Tensor] = average_gradients(grads_per_data)
        # print(len(avg_grads))

        for param, avg_grad in zip(model.parameters(), avg_grads):
            if param.grad is not None:
                param.grad = avg_grad

        # optimizer.zero_grad()
        # output: torch.Tensor = model(data)
        # losses: torch.Tensor = F.nll_loss(output, target, reduction='none')
        # print(losses)
        # print(len(data))
        # for loss in losses:
        #     loss.backward()
        #     # 勾配のクリッピング
        #     clip_gradients(model)
        # # 計算後の勾配を保存
        # grads: list[torch.Tensor] = [
        #     param.grad.clone()
        #     for param in model.parameters()
        #     if param.grad is not None
        # ]


        # ノイズの追加
        # noised_grads: list[torch.Tensor] = []
        # privacy_params: PrivacyParams = model.privacy_params

        # for grad in clipped_grads:
        #     if is_Scaling:
        #         noise: torch.Tensor = torch.distributions.Laplace(
        #             privacy_params.mean, privacy_params.var
        #         ).sample(grad.shape) / math.sqrt(len(grads))
        #     else:
        #         noise: torch.Tensor = torch.distributions.Laplace(
        #             privacy_params.mean, privacy_params.var
        #         ).sample(grad.shape)
        #     noised_grads.append(grad + noise)

        # オプティマイザのパラメータに手動で勾配を設定
        # for param, grad in zip(model.parameters(), noised_grads):
        #     param.grad = grad.to(device)

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
    test_loss: float = 0
    correct: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
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

def plot_graph(
    args,
    eps_values: list[float],
    all_correct_lists_without_scaling: list[list[int]],
    all_correct_lists_with_scaling: list[list[int]],
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
            marker="o",
            label="Without Scaling",
        )
        axes[i].plot(
            range(1, args.epochs + 1),
            noDP_list,
            marker="o",
            label="With Scaling",
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
        axes[i].set_title(f"eps={eps:.1f}")
        axes[i].legend()
        axes[i].grid()

    plt.suptitle(
        r"Impact of Laplace Mechanism for DP on Model Accuracy with $\sqrt{m}$ Scaling"
    )

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

    eps_values: list[float] = [0.1 * i for i in range(1, 3)]
    # eps_values: list[float] = [0.1 * i for i in range(1, 11)]
    all_correct_lists_without_scaling: list[list[int]] = []
    all_correct_lists_with_scaling: list[list[int]] = []

    for eps in eps_values:
        print(eps)
        privacy_params = PrivacyParams(eps=eps, sensitivity=1.0, mean=0)
        model_without_scaling: Net = Net(privacy_params).to(device)
        model_with_scaling: Net = Net(privacy_params).to(device)
        optimizer_without_scaling = optim.Adadelta(
            model_without_scaling.parameters(), lr=args.lr
        )
        optimizer_with_scaling = optim.Adadelta(
            model_with_scaling.parameters(), lr=args.lr
        )

        scheduler_without_scaling = StepLR(
            optimizer_without_scaling, step_size=1, gamma=args.gamma
        )
        scheduler_with_scaling = StepLR(
            optimizer_with_scaling, step_size=1, gamma=args.gamma
        )

        correct_list_without_scaling: list[int] = []
        correct_list_with_scaling: list[int] = []

        for epoch in range(1, args.epochs + 1):
            train(
                args,
                model_without_scaling,
                device,
                train_loader,
                optimizer_without_scaling,
                epoch,
                False,
            )
            correct: int = test(model_without_scaling, device, test_loader)
            correct_list_without_scaling.append(correct)
            scheduler_without_scaling.step()
            train(
                args,
                model_with_scaling,
                device,
                train_loader,
                optimizer_with_scaling,
                epoch,
                True,
            )
            correct = test(model_with_scaling, device, test_loader)
            correct_list_with_scaling.append(correct)
            scheduler_with_scaling.step()

        all_correct_lists_without_scaling.append(correct_list_without_scaling)
        all_correct_lists_with_scaling.append(correct_list_with_scaling)

    if args.save_model:
        torch.save(model_without_scaling.state_dict(), "mnist_cnn1.pt")
        torch.save(model_with_scaling.state_dict(), "mnist_cnn2.pt")

    plot_graph(
        args,
        eps_values,
        all_correct_lists_without_scaling,
        all_correct_lists_with_scaling,
    )


if __name__ == "__main__":
    main()
