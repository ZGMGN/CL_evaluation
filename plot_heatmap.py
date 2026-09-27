import os
import torch
import matplotlib.pyplot as plt

SIMILARITY_DIR = "similarity"
OUTPUT_DIR = "heatmap"

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


for layer in LAYERS:

    path = os.path.join(
        SIMILARITY_DIR,
        f"cosine_similarity_layer{layer}.pt"
    )

    similarity = torch.load(
        path,
        map_location="cpu"
    ).float()

    similarity = similarity.numpy()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        similarity,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Cosine Similarity")

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
    plt.title(f"Cosine Similarity - Layer {layer}")

    # 在格子里面显示具体数值
    for i in range(len(DATA_NAME)):
        for j in range(len(DATA_NAME)):
            plt.text(
                j,
                i,
                f"{similarity[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"cosine_similarity_layer{layer}.png"
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