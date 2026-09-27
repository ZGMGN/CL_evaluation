import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ACTIVATION_DIR = "activation"
OUTPUT_DIR = "heatmap_single_sample"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_NAME = [
    'AGNEWS',
    'AMAZON',
    'DBPEDIA',
    'IMDB',
    'MEDDIA',
    'QQP',
    'SST2',
    'TLDR',
    'WMT',
    'YELP'
]

LAYERS = [8, 12, 16, 20, 24, 28, 32]

# 取第几个 sample
SAMPLE_INDEX = 0


for layer in LAYERS:

    print()
    print("=" * 60)
    print(f"Layer {layer}")
    print("=" * 60)

    sample_vectors = []

    for name in DATA_NAME:

        path = os.path.join(
            ACTIVATION_DIR,
            f"{name}_layer{layer}_samples.pt"
        )

        vectors = torch.load(
            path,
            map_location="cpu"
        ).float()

        print(
            f"{name}: {vectors.shape}"
        )

        # 取一个 sample
        vector = vectors[SAMPLE_INDEX]

        sample_vectors.append(vector)

    # [10, 4096]
    matrix = torch.stack(sample_vectors)

    # L2 normalize
    matrix = F.normalize(
        matrix,
        p=2,
        dim=1
    )

    # [10, 10]
    similarity = matrix @ matrix.T

    print()
    print("Similarity:")
    print(similarity)

    # ==========================
    # Plot heatmap
    # ==========================

    similarity_np = similarity.numpy()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        similarity_np,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Cosine Similarity"
    )

    plt.xticks(
        range(len(DATA_NAME)),
        DATA_NAME,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(DATA_NAME)),
        DATA_NAME
    )

    plt.xlabel("Dataset")
    plt.ylabel("Dataset")

    plt.title(
        f"Single-Sample Cosine Similarity - Layer {layer}"
    )

    # 显示数字
    for i in range(len(DATA_NAME)):
        for j in range(len(DATA_NAME)):

            plt.text(
                j,
                i,
                f"{similarity_np[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"single_sample_similarity_layer{layer}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


print()
print("All heatmaps finished.")